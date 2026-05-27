import os, subprocess
DATASETS = {
    "llava_instruct_150k.json": "https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/llava_instruct_150k.json",
    "empathetic_dialogues_train.csv": "https://huggingface.co/datasets/facebook/empathetic_dialogues/resolve/main/empathetic_dialogues/train.csv",
    "indicvoices_hindi_train.csv": "https://huggingface.co/datasets/ai4bharat/indicvoices_r/resolve/main/Hindi/train.csv",
    "bhaav_dataset.csv": "https://raw.githubusercontent.com/sahil702/BHAAV/master/BHAAV-Dataset.csv",
    "dakshina_hi_romanized.tsv": "https://raw.githubusercontent.com/google-research-datasets/dakshina/master/hi/romanized/hi.translit.sampled.train.tsv"
}
for filename, url in DATASETS.items():
    print(f"Downloading {filename}...")
    subprocess.run(["curl", "-L", url, "-o", filename], check=True)
    print(f"Uploading to Modal...")
    subprocess.run(["modal", "volume", "put", "gemma4-sft-volume", filename, f"/data/{filename}"], check=True)
    os.remove(filename)
print("Volume is Ready!")
