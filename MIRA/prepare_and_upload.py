
import modal
import os

app = modal.App("dataset-processor")
volume = modal.Volume.from_name("gemma4-sft-volume")

@app.function(
    image=modal.Image.debian_slim().pip_install("pandas", "pyarrow", "openpyxl"),
    volumes={"/data": volume},
    timeout=3600
)
def process_remaining():
    import pandas as pd
    import json
    import glob
    
    print("Processing Remaining Datasets (Rasa, IndicTTS)...")
    
    # helper
    def save_jsonl(data, filename):
        if not data: return
        pd.DataFrame(data).to_json(f"/data/data/{filename}", orient="records", lines=True)
        print(f"Saved {len(data)} examples to {filename}")

    # 1. Rasa (Transcription based on columns)
    path = "/data/data/rasa.parquet"
    if os.path.exists(path):
        df = pd.read_parquet(path)
        if 'text' in df.columns:
            prompts = [
                "Ira, transcribe this for me:",
                "Can you write down what was said here?",
                "Listen and transcribe:",
                "Transcription task:",
                "What's being said in this audio?"
            ]
            processed = []
            for i, (_, row) in enumerate(df.iterrows()):
                txt = str(row['text'])
                if txt:
                    p = prompts[i % len(prompts)]
                    processed.append({"messages": [{"role": "user", "content": p}, {"role": "assistant", "content": txt}], "source": "rasa"})
            save_jsonl(processed, "rasa_processed.jsonl")

    # 2. IndicTTS (txt.done.data format)
    meta_path = "/data/data/indictts/txt.done.data"
    if os.path.exists(meta_path):
        print(f"Parsing IndicTTS meta: {meta_path}")
        processed = []
        tts_prompts = [
            "Convert this text to speech style:",
            "How would you say this out loud?",
            "Read this for me, Ira:",
            "TTS Text:",
            "Voice this content:"
        ]
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if line.startswith("(") and line.endswith(")"):
                        parts = line[1:-1].split('"', 2)
                        if len(parts) >= 2:
                            text = parts[1].strip()
                            if text:
                                p = tts_prompts[i % len(tts_prompts)]
                                processed.append({"messages": [{"role": "user", "content": p}, {"role": "assistant", "content": text}], "source": "indictts"})
        except Exception as e:
            print(f"Error parsing IndicTTS: {e}")
        save_jsonl(processed, "indictts_processed.jsonl")

    volume.commit()

if __name__ == "__main__":
    with app.run():
        process_remaining.remote()
