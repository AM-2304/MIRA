import modal, os, io

os.environ["HF_HOME"] = "/data/huggingface"
os.environ["COQUI_TOS_AGREED"] = "1"

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "curl", "wget", "ffmpeg", "libsndfile1", "espeak-ng")
    .pip_install(
        "torch==2.4.0", "torchaudio==2.4.0",
        extra_index_url="https://download.pytorch.org/whl/cu121"
    )
    .pip_install("transformers==4.33.3")
    .pip_install("TTS==0.22.0")
    .pip_install("fastapi", "uvicorn[standard]", "pydantic", "soundfile", "numpy")
    .run_commands(
        "COQUI_TOS_AGREED=1 python3 -c \"from TTS.api import TTS; _ = TTS('tts_models/multilingual/multi-dataset/xtts_v2')\""
    )
)

app = modal.App("ira-tts-service")
volume = modal.Volume.from_name("gemma4-sft-volume")

import re

# Exclusive Hinglish vs English word lists for classification
HINGLISH_WORDS = {
    "tum", "aap", "yeh", "woh", "mera", "meri", "mere", "tumhara", "tumhari", "tumhare",
    "kya", "kab", "kahan", "kyun", "kyu", "kaise", "kaisi", "hoon", "hai", "hain",
    "tha", "thi", "raha", "rahi", "rahe", "gaya", "gayi", "gaye", "yaar", "theek",
    "achha", "acha", "haan", "bas", "phir", "fir", "bohot", "bahut", "kuch", "nhi",
    "haina", "karna", "karta", "karti", "karte"
}

ENGLISH_WORDS = {
    "the", "to", "of", "and", "a", "in", "is", "you", "that", "it", "he", "was",
    "for", "on", "are", "as", "with", "his", "they", "i", "at", "be", "this",
    "have", "from", "what", "where", "why", "how", "who", "which", "them", "their",
    "she", "her", "him"
}

def is_hinglish(text: str) -> bool:
    # 1. Contains Devanagari characters (Hindi script)
    if any('\u0900' <= c <= '\u097F' for c in text):
        return True
    
    # 2. Compare Hinglish vs English word counts for Latin script
    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    if not words:
        return False
    
    hinglish_count = sum(1 for w in words if w in HINGLISH_WORDS)
    english_count = sum(1 for w in words if w in ENGLISH_WORDS)
    
    return hinglish_count > english_count


@app.cls(
    image=image,
    gpu="A10G",
    volumes={"/data": volume},
    scaledown_window=600,
    min_containers=0,  # Scales down to 0 after 10 minutes of inactivity to conserve GPU credits!
)
class TTSService:
    @modal.enter()
    def load_model(self):
        """
        Natively load base model through the official high-level TTS wrapper,
        then inject the fine-tuned GPT weights using strict=False merge.
        This provides beautiful sentence splitting and text normalization out-of-the-box!
        """
        import os, glob
        os.environ["COQUI_TOS_AGREED"] = "1"
        import torch
        from TTS.api import TTS

        print("[TTS] Initializing high-level TTS API wrapper...")
        # Loads from pre-cached baked container image instantly!
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

        # Check if fine-tuned model exists in the volume
        fine_tuned_dir = "/data/finetuned_voice"
        use_fine_tuned = os.path.exists(os.path.join(fine_tuned_dir, "model.pth"))

        if use_fine_tuned:
            print("[TTS] Loading custom fine-tuned GPT weights into high-level synthesizer...")
            checkpoint = torch.load(os.path.join(fine_tuned_dir, "model.pth"), map_location="cpu")
            state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
            
            # Filter state dict keys to prevent size mismatches
            model_dict = self.tts.synthesizer.tts_model.state_dict()
            filtered_state_dict = {
                k: v for k, v in state_dict.items()
                if k in model_dict and v.size() == model_dict[k].size()
            }
            
            self.tts.synthesizer.tts_model.load_state_dict(filtered_state_dict, strict=False)
            self.ref_wav = os.path.join(fine_tuned_dir, "reference.wav")
            print("[TTS] Fine-tuned speaker GPT weights loaded into high-level synthesizer!")
        else:
            print("[TTS] No fine-tuned model found. Using default Hinglish voice reference...")
            self.ref_wav = "/data/ref_hinglish.wav"

        print("[TTS] XTTS-v2 high-level API loaded successfully!")

        # Warmup GPU to pre-compile CUDA kernels
        print("[TTS] Running warmup inference...")
        try:
            self.tts.tts(
                text="Haan.",
                speaker_wav=self.ref_wav,
                language="hi"
            )
            print("[TTS] Warmup done — container is fully hot and ready.")
        except Exception as e:
            print(f"[TTS] Warmup failed (non-fatal): {e}")

    @modal.asgi_app()
    def web_app(self):
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import Response
        from pydantic import BaseModel
        import soundfile as sf, numpy as np

        api = FastAPI(title="Mira TTS — XTTS-v2 (High-Level Weight Override)")

        class TTSRequest(BaseModel):
            text: str

        @api.post("/speak")
        async def speak(req: TTSRequest):
            text = req.text.strip()
            if not text:
                raise HTTPException(400, "Empty text")

            # Auto-detect language while always using the fine-tuned reference voice.
            # Devanagari script -> 'hi' | Pure English -> 'en' | Hinglish (mixed) -> 'hi'
            # XTTS-v2 voice cloning via speaker_wav preserves the Indian female voice
            # identity regardless of which language is selected.
            import re
            has_devanagari = bool(re.search(r'[\u0900-\u097F]', text))
            hinglish_markers = ["main","tum","aap","hai","hain","kya","yaar","nahi","hoon","kar","raha","rahi","tha","thi","aur","bas","haan","nhi","mera","teri"]
            words_lower = text.lower().split()
            is_hinglish = sum(1 for w in words_lower if w in hinglish_markers) >= 1
            lang = "hi" if (has_devanagari or is_hinglish) else "en"
            print(f"[TTS] Language detected: '{lang}' | {text[:60]}")

            try:
                wav = self.tts.tts(
                    text=text,
                    speaker_wav=self.ref_wav,
                    language=lang,
                    temperature=0.75,
                    repetition_penalty=2.0,
                    length_penalty=1.0,
                )
                wav_arr = np.array(wav, dtype=np.float32)
                buf = io.BytesIO()
                sf.write(buf, wav_arr, 24000, format="WAV")
                buf.seek(0)
                return Response(content=buf.read(), media_type="audio/wav")

            except Exception as e:
                import traceback; traceback.print_exc()
                raise HTTPException(500, str(e))

        @api.get("/health")
        async def health():
            return {"status": "healthy", "model": "xtts-v2-highlevel-override"}

        return api

