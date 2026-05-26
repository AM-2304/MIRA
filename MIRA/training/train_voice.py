import modal
import os

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
    .env({"COQUI_TOS_AGREED": "1"})
    .add_local_file("training/voice samples.wav", "/root/voice_samples.wav")
)

app = modal.App("ira-tts-training")
volume = modal.Volume.from_name("gemma4-sft-volume")

@app.function(
    image=image,
    gpu="A10G",
    timeout=18000,
    volumes={"/data": volume},
)
def train_voice():
    import os
    os.environ["COQUI_TOS_AGREED"] = "1"
    import glob
    import shutil
    import soundfile as sf
    import numpy as np
    from transformers import pipeline

    # 1. Segment raw audio
    print("[1/5] Segmenting raw training audio file /root/voice_samples.wav...")
    os.makedirs("/root/dataset/wavs", exist_ok=True)
    
    data, sr = sf.read("/root/voice_samples.wav")
    if len(data.shape) > 1:
        data = np.mean(data, axis=1) # Mono conversion
        
    # Determine silence threshold via RMS of 50ms windows
    window_size = int(sr * 0.05)
    num_windows = len(data) // window_size
    rms = np.zeros(num_windows)
    for i in range(num_windows):
        rms[i] = np.sqrt(np.mean(data[i*window_size:(i+1)*window_size]**2))
    
    threshold = np.percentile(rms, 15) # 15th percentile is quiet noise floor
    threshold = max(threshold, 0.005) # Keep a safe minimum
    
    # Group windows into voice segments
    is_voice = rms > threshold
    segments = []
    in_segment = False
    start_idx = 0
    
    for i, val in enumerate(is_voice):
        if val and not in_segment:
            start_idx = i * window_size
            in_segment = True
        elif not val and in_segment:
            end_idx = i * window_size
            duration = (end_idx - start_idx) / sr
            if 2.0 <= duration <= 12.0:
                segments.append((start_idx, end_idx))
            in_segment = False
            
    # Save segments
    saved_files = []
    for i, (start, end) in enumerate(segments):
        out_path = f"/root/dataset/wavs/segment_{i:04d}.wav"
        sf.write(out_path, data[start:end], sr)
        saved_files.append(out_path)

    # Fallback to uniform chunks if VAD produced too few segments
    if len(saved_files) < 50:
        print(f"VAD only produced {len(saved_files)} segments. Falling back to uniform 8-second chunking to ensure full coverage...")
        shutil.rmtree("/root/dataset/wavs")
        os.makedirs("/root/dataset/wavs", exist_ok=True)
        saved_files = []
        chunk_samples = int(sr * 8.0)
        for i in range(0, len(data), chunk_samples):
            end = min(i + chunk_samples, len(data))
            if (end - i) >= sr * 2.0: # at least 2s
                out_path = f"/root/dataset/wavs/segment_{i // chunk_samples:04d}.wav"
                sf.write(out_path, data[i:end], sr)
                saved_files.append(out_path)
        print(f"Created {len(saved_files)} uniform 8.0s audio chunks.")
    else:
        print(f"Created {len(saved_files)} audio segments of 2.0s to 12.0s duration.")

    # 2. Transcribe using Whisper
    print("[2/5] Auto-transcribing audio segments using Whisper-base on GPU...")
    whisper = pipeline("automatic-speech-recognition", model="openai/whisper-base", device=0)
    
    metadata_lines = []
    for i, file_path in enumerate(saved_files):
        try:
            result = whisper(file_path)
            text = result["text"].strip()
            if text:
                rel_path = f"segment_{i:04d}"
                metadata_lines.append(f"{rel_path}|{text}|{text}")
                if i % 10 == 0 or i == len(saved_files) - 1:
                    print(f"  [{i + 1}/{len(saved_files)}] Transcribed: {text[:60]}")
        except Exception as ex:
            print(f"  Skipping segment {i} due to Whisper error: {ex}")
            
    with open("/root/dataset/metadata.csv", "w") as f:
        f.write("\n".join(metadata_lines) + "\n")
    print(f"Dataset manifest created with {len(metadata_lines)} transcribed clips.")

    # 3. Setup Coqui model
    print("[3/5] Setting up Coqui XTTS v2 fine-tuning config...")
    from TTS.api import TTS as _TTS
    _loader = _TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    
    cache_dirs = glob.glob(os.path.expanduser(
        "~/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2*"
    ))
    if not cache_dirs:
        cache_dirs = glob.glob("/root/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2*")
    ckpt_dir = cache_dirs[0] if cache_dirs else None
    print(f"Base model loaded from: {ckpt_dir}")
    if ckpt_dir:
        print("Files inside ckpt_dir:", os.listdir(ckpt_dir))
        
        # Download required dvae.pth and mel_stats.pth from Hugging Face if they are missing
        import urllib.request
        def download_file(url, filename):
            filepath = os.path.join(ckpt_dir, filename)
            if not os.path.exists(filepath):
                print(f"Downloading {filename} from {url}...")
                urllib.request.urlretrieve(url, filepath)
                print(f"{filename} downloaded successfully!")
            else:
                print(f"{filename} already exists in {ckpt_dir}.")

        download_file("https://huggingface.co/coqui/XTTS-v2/resolve/main/dvae.pth", "dvae.pth")
        download_file("https://huggingface.co/coqui/XTTS-v2/resolve/main/mel_stats.pth", "mel_stats.pth")
        print("Updated files inside ckpt_dir:", os.listdir(ckpt_dir))

    from trainer import Trainer, TrainerArgs
    from TTS.config.shared_configs import BaseDatasetConfig
    from TTS.tts.datasets import load_tts_samples
    from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig

    # Configure audio using dvae from cached directory
    audio_config = XttsAudioConfig(sample_rate=22050)

    # Configure model arguments
    model_args = GPTArgs(
        max_conditioning_length=132300,  # 6s
        min_conditioning_length=11025,   # 0.5s
        debug_loading_failures=False,
        max_wav_length=255995,           # ~11.6s
        max_text_length=200,
        mel_norm_file=os.path.join(ckpt_dir, "mel_stats.pth"),
        dvae_checkpoint=os.path.join(ckpt_dir, "dvae.pth"),
        xtts_checkpoint=os.path.join(ckpt_dir, "model.pth"),
        tokenizer_file=os.path.join(ckpt_dir, "vocab.json"),
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
    )

    # Dataset configuration in LJSpeech format
    dataset_config = BaseDatasetConfig(
        formatter="ljspeech",
        dataset_name="ljspeech",
        path="/root/dataset",
        meta_file_train="metadata.csv",
        language="en",
    )

    train_samples, eval_samples = load_tts_samples(dataset_config, eval_split=False)

    # Fine-tuning config
    config = GPTTrainerConfig(
        output_path="/root/training_output",
        model_args=model_args,
        audio=audio_config,
        batch_size=4,
        run_eval=False,
        test_sentences_file=None,
        epochs=10,
        lr=5e-6,
        save_step=100,
        print_step=10,
        optimizer="AdamW",
        optimizer_params={"betas": (0.9, 0.96), "weight_decay": 0.01},
    )

    # 4. Train
    print("[4/5] Initializing GPTTrainer and starting fine-tuning loop...")
    model = GPTTrainer.init_from_config(config)
    trainer = Trainer(
        TrainerArgs(grad_accum_steps=64),
        config,
        output_path="/root/training_output",
        model=model,
        train_samples=train_samples,
        eval_samples=None,
    )
    trainer.fit()

    # 5. Export
    print("[5/5] Extracting fine-tuned weights and exporting to persistent volume...")
    trained_ckpts = glob.glob("/root/training_output/**/best_model.pth", recursive=True)
    if not trained_ckpts:
        trained_ckpts = glob.glob("/root/training_output/**/*.pth", recursive=True)
    
    if not trained_ckpts:
        raise RuntimeError("Fine-tuning finished but no checkpoint .pth was found in output folder!")
    
    best_model_path = trained_ckpts[0]
    print(f"Fine-tuned model checkpoint found at: {best_model_path}")

    dest_dir = "/data/finetuned_voice"
    os.makedirs(dest_dir, exist_ok=True)

    # Copy fine-tuned weights and configuration files
    shutil.copy(best_model_path, os.path.join(dest_dir, "model.pth"))
    shutil.copy(os.path.join(ckpt_dir, "vocab.json"), os.path.join(dest_dir, "vocab.json"))
    shutil.copy(os.path.join(ckpt_dir, "dvae.pth"), os.path.join(dest_dir, "dvae.pth"))
    shutil.copy(os.path.join(ckpt_dir, "mel_stats.pth"), os.path.join(dest_dir, "mel_norms.pth"))
    shutil.copy(os.path.join(ckpt_dir, "mel_stats.pth"), os.path.join(dest_dir, "mel_stats.pth"))
    shutil.copy(os.path.join(ckpt_dir, "config.json"), os.path.join(dest_dir, "config.json"))

    # Deploy a clean segment to be the conditioning reference audio
    shutil.copy(saved_files[0], os.path.join(dest_dir, "reference.wav"))
    
    volume.commit()
    print(f"\nTraining successfully complete! Checkpoint deployed to {dest_dir} on persistent volume.")
