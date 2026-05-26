import modal

volume = modal.Volume.from_name("gemma4-sft-volume")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .run_commands(
        "pip install torch torchvision "
        "'unsloth_zoo @ git+https://github.com/unslothai/unsloth-zoo.git' "
        "'unsloth @ git+https://github.com/unslothai/unsloth.git' "
        "datasets>=3.0.0 trl>=0.15.0 peft accelerate bitsandbytes "
        "pandas transformers==5.5.0"
    )
)
app = modal.App(name="ira-brain-merge", image=image)

@app.function(
    gpu="A100-80GB",
    volumes={"/data": volume},
    timeout=3600,
)
def merge_model():
    import torch
    from unsloth import FastLanguageModel
    import json
    import os
    import shutil

    sft_path = "/data/ira_sft_v1/checkpoint-22191"
    dpo_path = "/data/ira_dpo_v2/dpo_checkpoint"
    temp_sft_merged = "/data/ira_sft_merged_temp"
    final_path = "/data/ira_brain_merged"

    # 1. Create SFT-merged base
    print("Step 1: Fusing SFT into Base Model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=sft_path,
        max_seq_length=4096,
        load_in_4bit=False,
        dtype=torch.bfloat16,
    )
    model.save_pretrained_merged(temp_sft_merged, tokenizer, save_method="merged_16bit")
    print(f"SFT-merged base saved to {temp_sft_merged}")
    
    # 2. Update DPO adapter_config.json to point to the new base
    print("Step 2: Patching DPO config to use SFT-merged base...")
    config_path = os.path.join(dpo_path, "adapter_config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    
    config["base_model_name_or_path"] = temp_sft_merged
    
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print("DPO config patched.")

    # 3. Load DPO with the new base and merge
    print("Step 3: Fusing DPO into the patched base...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=dpo_path,
        max_seq_length=4096,
        load_in_4bit=False,
        dtype=torch.bfloat16,
    )
    model.save_pretrained_merged(final_path, tokenizer, save_method="merged_16bit")
    
    volume.commit()
    print(f"Merge complete! Final brain at: {final_path}")

@app.local_entrypoint()
def main():
    merge_model.remote()
