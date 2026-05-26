import modal, os, io

os.environ["HF_HOME"] = "/data/huggingface"

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "curl", "wget", "ffmpeg", "libsndfile1")
    .pip_install(
        "torch==2.4.0",
        extra_index_url="https://download.pytorch.org/whl/cu121"
    )
    .pip_install("faster-whisper")
    .pip_install("fastapi", "uvicorn[standard]", "pydantic", "soundfile", "numpy", "python-multipart")
)

app = modal.App("ira-stt-service")
volume = modal.Volume.from_name("gemma4-sft-volume")


@app.cls(
    image=image,
    gpu="A10G",
    volumes={"/data": volume},
    scaledown_window=300,
    min_containers=1,  # Keeps Faster-Whisper GPU container alive 24/7 for zero cold starts!
)
class STTService:
    @modal.enter()
    def load_model(self):
        """
        Pre-load faster-whisper on GPU to enable extremely low-latency transcription.
        Using the highly optimized large-v3-turbo model for best-in-class multi-language/Hinglish accuracy.
        """
        from faster_whisper import WhisperModel
        print("[STT] Loading Faster-Whisper model (large-v3-turbo) on CUDA...")
        # Under CUDA, we use float16 precision for maximum acceleration on A10G
        self.model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
        print("[STT] Model loaded successfully.")

    @modal.asgi_app()
    def web_app(self):
        from fastapi import FastAPI, HTTPException, UploadFile, File
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel
        import numpy as np
        import soundfile as sf
 
        api = FastAPI(title="Mira STT — Faster-Whisper (CUDA)")
        api.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @api.post("/transcribe")
        async def transcribe(file: UploadFile = File(...)):
            """
            Accepts raw audio file upload, decodes it, transcribes it,
            and returns high-fidelity text.
            """
            content = await file.read()
            if not content:
                raise HTTPException(400, "Empty audio file")

            try:
                import subprocess
                
                # Convert input stream using ffmpeg to a standard WAV stream in memory
                process = subprocess.Popen(
                    [
                        "ffmpeg",
                        "-i", "pipe:0",
                        "-f", "wav",
                        "-acodec", "pcm_s16le",
                        "-ar", "16000",
                        "-ac", "1",
                        "pipe:1"
                    ],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                stdout_data, stderr_data = process.communicate(input=content)
                
                if process.returncode != 0:
                    err_msg = stderr_data.decode("utf-8", errors="ignore")
                    print(f"[STT] ffmpeg failed: {err_msg}")
                    raise HTTPException(500, f"ffmpeg conversion failed: {err_msg}")
                
                # Read WAV from memory
                audio_file = io.BytesIO(stdout_data)
                audio_data, sample_rate = sf.read(audio_file)
                audio_data = audio_data.astype(np.float32)

                print(f"[STT] Transcribing {len(audio_data)/sample_rate:.2f}s of audio...")
                
                # Transcribe with language auto-detect to allow clean multilingual/Hinglish support
                segments, info = self.model.transcribe(
                    audio_data,
                    beam_size=5,
                    language=None, # Auto-detect language dynamically (extremely robust!)
                    initial_prompt="Yaar, kya chal raha hai? How is the vibe?", # Hinglish priming
                )

                text = " ".join([seg.text for seg in segments]).strip()
                print(f"[STT] Result: \"{text}\" (detected lang: {info.language} with prob {info.language_probability:.2f})")

                return {
                    "text": text,
                    "language": info.language,
                    "probability": info.language_probability
                }

            except Exception as e:
                import traceback; traceback.print_exc()
                raise HTTPException(500, f"STT inference failed: {str(e)}")

        @api.get("/health")
        async def health():
            return {"status": "healthy", "model": "faster-whisper-large-v3-turbo"}

        return api
