import modal, os, io, time, re, math, asyncio
import numpy as np

os.environ["HF_HOME"] = "/data/huggingface"

# Ultra-lightweight image to keep the WebSocket gateway fast and highly responsive
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install("fastapi", "uvicorn[standard]", "soundfile", "numpy", "requests", "aiohttp")
)

app = modal.App("ira-gateway-service")
volume = modal.Volume.from_name("gemma4-sft-volume")


# ── Prosody Vibe Analyzer (Speech Signal Processing) ────────────────────────
def analyze_vocal_prosody(audio_data: np.ndarray, sample_rate: int) -> dict:
    """
    Analyzes raw audio PCM array to extract key vocal elements:
    - speech_rate: estimated syllables/second via envelope peak counting
    - pitch_std: standard deviation of pitch frequency (monotone vs modulated)
    - mean_energy: average signal energy (soft whisper vs loud shouting)
    """
    if len(audio_data) == 0:
        return {"energy": "normal", "tempo": "normal", "inflection": "normal", "description": "unspecified"}
        
    # 1. Energy Analysis (Volume / Amplitude RMS)
    rms = np.sqrt(np.mean(audio_data**2))
    
    if rms < 0.005:
        energy_state = "soft"
        energy_desc = "speaks softly, almost in a gentle whisper"
    elif rms > 0.12:
        energy_state = "loud"
        energy_desc = "speaks loudly, with highly energetic delivery"
    else:
        energy_state = "normal"
        energy_desc = "speaks at a natural, balanced volume"
        
    # 2. Pitch Inflection Analysis (using Autocorrelation to estimate F0 in voiced regions)
    # Target voice F0 range: 60Hz to 400Hz
    # At 16000Hz sample rate, that corresponds to lags from 40 to 266 samples
    pitch_frequencies = []
    frame_size = int(0.03 * sample_rate) # 30ms window
    hop_size = int(0.015 * sample_rate) # 15ms step
    
    for start in range(0, len(audio_data) - frame_size, hop_size):
        frame = audio_data[start:start+frame_size]
        # Skip silent frames (energy threshold)
        if np.std(frame) < 0.003:
            continue
            
        # Autocorrelation
        corr = np.correlate(frame, frame, mode='full')
        corr = corr[len(corr)//2:] # Keep positive lags
        
        min_lag = int(sample_rate / 400) # ~40
        max_lag = int(sample_rate / 60)  # ~266
        
        if len(corr) > max_lag:
            lag_peak = np.argmax(corr[min_lag:max_lag]) + min_lag
            if corr[lag_peak] > 0.3 * corr[0]: # Voicing threshold
                f0 = sample_rate / lag_peak
                pitch_frequencies.append(f0)
                
    if pitch_frequencies:
        pitch_std = np.std(pitch_frequencies)
        pitch_mean = np.mean(pitch_frequencies)
        
        if pitch_std < 18:
            inflection = "flat"
            inflection_desc = "has a slightly monotone or flat voice (suggesting tiredness, sadness, or calm reserve)"
        elif pitch_std > 48:
            inflection = "modulated"
            inflection_desc = "has a highly expressive, animated, and modulated tone (suggesting playfulness, excitement, or surprise)"
        else:
            inflection = "normal"
            inflection_desc = "has a naturally warm, balanced pitch inflection"
    else:
        pitch_std = 0
        pitch_mean = 0
        inflection = "normal"
        inflection_desc = "speaks in an unspecified tone"
        
    # 3. Speech Rate (Tempo estimation via Short-Time Energy envelope peak detection)
    # Compute energy envelope
    window_len = int(0.08 * sample_rate) # 80ms energy smoothing
    envelope = np.convolve(np.abs(audio_data), np.ones(window_len)/window_len, mode='same')
    
    # Peak counting for syllables
    peaks = 0
    threshold = np.mean(envelope) * 1.15
    for i in range(1, len(envelope)-1):
        if envelope[i] > envelope[i-1] and envelope[i] > envelope[i+1] and envelope[i] > threshold:
            peaks += 1
            
    duration = len(audio_data) / sample_rate
    rate = peaks / duration if duration > 0 else 0
    
    if rate < 2.0:
        tempo = "slow"
        tempo_desc = "speaks at a slow, deliberate, or hesitant pace (often indicating reflection, melancholy, or fatigue)"
    elif rate > 3.8:
        tempo = "fast"
        tempo_desc = "speaks at a rapid, fast-paced tempo (often indicating high energy, excitement, or mild anxiety)"
    else:
        tempo = "normal"
        tempo_desc = "speaks at a moderate, natural tempo"
        
    # Construct complete descriptive vocal vibe sentence
    description = f"[User's voice context: The user {energy_desc}, {tempo_desc}, and {inflection_desc}.]"
    
    return {
        "energy": energy_state,
        "tempo": tempo,
        "inflection": inflection,
        "pitch_mean": pitch_mean,
        "pitch_std": pitch_std,
        "speech_rate": rate,
        "description": description
    }


# ── Sentence Boundary Splitter ──────────────────────────────────────────────
def extract_sentences(text: str, finished: bool = False):
    """
    Statefully extracts complete sentences or clauses from streaming text.
    Returns (complete_sentences, remaining_buffer).
    """
    # Hard boundaries: dot, question mark, exclamation, newline, comma (for rapid parsing)
    bounds = [".", "?", "!", "\n", ","]
    
    sentences = []
    buffer = text
    
    while True:
        boundary_idx = -1
        for b in bounds:
            idx = buffer.find(b)
            if idx != -1 and (boundary_idx == -1 or idx < boundary_idx):
                boundary_idx = idx
                
        if boundary_idx != -1:
            sentence = buffer[:boundary_idx + 1].strip()
            buffer = buffer[boundary_idx + 1:]
            if len(sentence) > 3:
                sentences.append(sentence)
        else:
            break
            
    if finished and buffer.strip():
        sentences.append(buffer.strip())
        buffer = ""
        
    return sentences, buffer


@app.cls(
    image=image,
    volumes={"/data": volume},
    scaledown_window=300,
    min_containers=1,  # Keeps Gateway container alive 24/7 for zero connection delay!
)
class GatewayService:
    @modal.enter()
    def setup_dirs(self):
        """Ensure audio storage logs are persistent."""
        os.makedirs("/data/audio_logs", exist_ok=True)
        print("[Gateway] Persistent audio storage directories created.")

    @modal.asgi_app()
    def web_app(self):
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect
        from fastapi.middleware.cors import CORSMiddleware
        import soundfile as sf
        import aiohttp
        import json

        api = FastAPI(title="Mira Orchestration Gateway")
        api.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        SGLANG_URL = "https://rumik-ai--ira-sglang-service-sglangservice-web-app.modal.run/v1/chat/completions"
        TTS_URL = "https://rumik-ai--ira-tts-service-ttsservice-web-app.modal.run/speak"
        STT_URL = "https://rumik-ai--ira-stt-service-sttservice-web-app.modal.run/transcribe"

        @api.websocket("/v1/sts/live")
        async def websocket_sts(ws: WebSocket):
            await ws.accept()
            print("[Gateway] Bidirectional WebSocket STS call initialized.")

            # Session-specific state variables
            session_id = f"sess_{int(time.time())}"
            audio_buffer = []  # Accumulates incoming PCM chunks
            is_recording = False
            last_audio_time = time.time()
            conversation_history = []

            # VAD variables (Energy threshold)
            SILENCE_THRESHOLD = 0.003
            SILENCE_DURATION_SECONDS = 0.8  # Wait 800ms of silence after voicing to trigger completion
            voiced_frames_count = 0
            silent_frames_count = 0

            async def process_user_turn():
                nonlocal audio_buffer, voiced_frames_count, silent_frames_count, conversation_history
                if not audio_buffer:
                    return

                # 1. Combine binary raw PCM data (Client streams float32 16kHz audio)
                try:
                    raw_pcm = np.concatenate(audio_buffer, axis=0)
                except Exception as e:
                    print(f"[Gateway] Error combining audio: {e}")
                    audio_buffer = []
                    return
                
                audio_buffer = [] # Reset buffer instantly
                voiced_frames_count = 0
                silent_frames_count = 0

                # Guard: Minimum 0.4s of audio to prevent accidental tap spikes from triggering SGLang
                if len(raw_pcm) < 16000 * 0.4:
                    print("[Gateway] Audio too short, discarding transient.")
                    return

                # 2. Save raw user audio permanently for auditing/continuity
                timestamp = int(time.time() * 1000)
                user_filename = f"/data/audio_logs/user_{session_id}_{timestamp}.wav"
                sf.write(user_filename, raw_pcm, 16000, format="WAV")
                print(f"[Gateway] Raw user audio preserved at {user_filename}")

                # 3. Parallel Tasks: STT Transcription & Vocal Prosody Analysis
                # We start the prosody analysis immediately in parallel since it runs on local CPU
                prosody_result = analyze_vocal_prosody(raw_pcm, 16000)
                vocal_vibe_note = prosody_result["description"]
                print(f"[Gateway] Prosody complete: {vocal_vibe_note}")

                # Notify client that transcription is starting
                await ws.send_json({"type": "status", "status": "transcribing"})

                # Upload raw WAV to STT Service
                stt_text = ""
                try:
                    async with aiohttp.ClientSession() as session:
                        with open(user_filename, "rb") as f:
                            data = aiohttp.FormData()
                            data.add_field("file", f, filename="audio.wav", content_type="audio/wav")
                            async with session.post(STT_URL, data=data) as resp:
                                if resp.status == 200:
                                    res = await resp.json()
                                    stt_text = res.get("text", "").strip()
                except Exception as e:
                    print(f"[Gateway] STT service request failed: {e}")

                if not stt_text:
                    print("[Gateway] STT returned empty result. Listening again...")
                    await ws.send_json({"type": "status", "status": "listening"})
                    return

                # Send verified transcript back to frontend to render in chat
                await ws.send_json({
                    "type": "transcript",
                    "sender": "user",
                    "text": stt_text,
                    "prosody": prosody_result
                })

                # 4. Formulate System Prompt with Prosody Vibe metadata injected
                # Append the tone/vibe directly so Gemma-4 knows if user is sad, energetic, whispering, etc.
                sglang_system_note = (
                    "You are Mira. Greet the user naturally. "
                    "Adapt your tone to their vocal state. "
                    f"VIBE CONTEXT: {vocal_vibe_note}"
                )

                # Formulate final message array
                # Incorporate user transcript
                conversation_history.append({"role": "user", "content": stt_text})

                # Notify client that Mira is processing the response
                await ws.send_json({"type": "status", "status": "thinking"})

                # 5. Pipeline SGLang LLM streaming directly into sentence chunking & TTS synthesis
                llm_response_buffer = ""
                sentence_buffer = ""
                tts_tasks = []

                async def generate_speech_for_chunk(text_chunk: str, chunk_index: int):
                    """Sends a sentence chunk to the TTS endpoint and streams the output to the client."""
                    print(f"[Gateway TTS] Synthesizing chunk {chunk_index}: \"{text_chunk}\"")
                    try:
                        async with aiohttp.ClientSession() as session:
                            async with session.post(TTS_URL, json={"text": text_chunk}) as resp:
                                if resp.status == 200:
                                    audio_bytes = await resp.read()
                                    # Send synthesized audio chunk to browser over WebSocket
                                    await ws.send_bytes(audio_bytes)
                                    print(f"[Gateway TTS] Streamed chunk {chunk_index} ({len(audio_bytes)} bytes)")
                                else:
                                    print(f"[Gateway TTS] Failed for chunk {chunk_index}: {resp.status}")
                    except Exception as ex:
                        print(f"[Gateway TTS] Error in task: {ex}")

                try:
                    payload = {
                        "model": "ira",
                        "messages": [
                            {"role": "system", "content": sglang_system_note},
                            *conversation_history[-8:] # Retain recent history
                        ],
                        "max_tokens": 256,
                        "temperature": 0.8,
                        "stream": True
                    }

                    chunk_count = 0
                    async with aiohttp.ClientSession() as session:
                        async with session.post(SGLANG_URL, json=payload) as resp:
                            if resp.status == 200:
                                # Read line-by-line streaming payload
                                async for line in resp.content:
                                    line_str = line.decode('utf-8').strip()
                                    if not line_str.startswith("data:"):
                                        continue
                                    if "[DONE]" in line_str:
                                        break
                                    
                                    try:
                                        data = json.loads(line_str[5:])
                                        token = data["choices"][0]["delta"].get("content", "")
                                        if token:
                                            llm_response_buffer += token
                                            sentence_buffer += token
                                            
                                            # Stream interim text to UI
                                            await ws.send_json({
                                                "type": "token",
                                                "sender": "mira",
                                                "text": token
                                            })

                                            # Chunk sentences on boundaries to pipeline to TTS
                                            complete_sentences, remaining = extract_sentences(sentence_buffer)
                                            sentence_buffer = remaining

                                            for s in complete_sentences:
                                                chunk_count += 1
                                                # Dispatch speech generation immediately (asynchronous pipelining)
                                                task = asyncio.create_task(generate_speech_for_chunk(s, chunk_count))
                                                tts_tasks.append(task)
                                    except Exception:
                                        pass

                    # Process remaining sentence chunk fragment
                    final_sentences, _ = extract_sentences(sentence_buffer, finished=True)
                    for s in final_sentences:
                        chunk_count += 1
                        task = asyncio.create_task(generate_speech_for_chunk(s, chunk_count))
                        tts_tasks.append(task)

                    # Wait for all speech segments to finish synthesizing/routing
                    if tts_tasks:
                        await asyncio.gather(*tts_tasks)

                    # Save complete Mira text reply to history
                    conversation_history.append({"role": "assistant", "content": llm_response_buffer})
                    print(f"[Gateway] Stream complete. Mira said: \"{llm_response_buffer}\"")

                except Exception as e:
                    print(f"[Gateway] LLM/TTS Pipeline failed: {e}")

                # Turn complete — listen again!
                await ws.send_json({"type": "status", "status": "listening"})

            try:
                while True:
                    # Receive client binary frames (PCM Float32 16kHz) or text signals
                    message = await ws.receive()
                    
                    if "bytes" in message:
                        # 1. Decode Raw Binary PCM Frame
                        pcm_chunk = np.frombuffer(message["bytes"], dtype=np.float32)
                        
                        if len(pcm_chunk) > 0:
                            audio_buffer.append(pcm_chunk)
                            last_audio_time = time.time()
                            
                            # 2. VAD: Analyze energy of current frame to detect silence
                            rms_frame = np.sqrt(np.mean(pcm_chunk**2))
                            if rms_frame > SILENCE_THRESHOLD:
                                voiced_frames_count += 1
                                silent_frames_count = 0
                                if not is_recording:
                                    is_recording = True
                                    print("[Gateway] Voice detected. Recording...")
                                    await ws.send_json({"type": "status", "status": "recording"})
                            else:
                                if is_recording:
                                    silent_frames_count += 1
                                    # Silence threshold met — user stopped speaking!
                                    silence_time = silent_frames_count * (len(pcm_chunk) / 16000)
                                    if silence_time >= SILENCE_DURATION_SECONDS:
                                        print("[Gateway] Silence detected. Triggering response...")
                                        is_recording = False
                                        # Run the complete user turn pipeline
                                        asyncio.create_task(process_user_turn())

                    elif "text" in message:
                        data = json.loads(message["text"])
                        # Handle text interruptions or settings
                        if data.get("type") == "interrupt":
                            print("[Gateway] Barge-in interrupt received from client. Purging pipeline.")
                            audio_buffer = []
                            is_recording = False
                            voiced_frames_count = 0
                            silent_frames_count = 0
                            await ws.send_json({"type": "status", "status": "listening"})
                            
            except WebSocketDisconnect:
                print("[Gateway] WebSocket STS disconnected.")
            except Exception as e:
                print(f"[Gateway] Unexpected error: {e}")

        return api
