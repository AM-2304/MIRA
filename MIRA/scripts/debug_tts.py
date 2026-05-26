import modal
from serve.serve_tts import image, volume

app = modal.App("debug-tts")

@app.function(
    image=image,
    gpu="A10G",
    volumes={"/data": volume},
)
def debug_load():
    import os, traceback
    print("--- DEBUG STARTING ---")
    try:
        os.environ["COQUI_TOS_AGREED"] = "1"
        from TTS.api import TTS as _TTS
        print("Imported Coqui TTS successfully.")
        
        print("Instantiating base model _TTS...")
        # Force cpu or gpu
        _loader = _TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        print("TTS loader instantiated successfully!")
        
        base_dir = _loader.synthesizer.tts_model.config.model_dir if hasattr(
            _loader.synthesizer, 'tts_model') else None
        print(f"base_dir: {base_dir}")
        
    except Exception as e:
        print("!!! EXCEPTION CAUGHT !!!")
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    with app.run():
        debug_load.remote()
