import modal
import os

app = modal.App(name="export-ira-model")
volume = modal.Volume.from_name("gemma4-sft-volume")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .run_commands(
        "pip install torch torchvision "
        "'unsloth_zoo @ git+https://github.com/unslothai/unsloth-zoo.git' "
        "'unsloth @ git+https://github.com/unslothai/unsloth.git' "
        "datasets>=3.0.0 trl>=0.15.0 peft accelerate bitsandbytes "
        "wandb pandas transformers==5.5.0"
    )
)

@app.function(image=image, gpu="A100", volumes={"/data": volume}, timeout=3600)
def export_model():
    from unsloth import FastLanguageModel
    import torch

    sft_path    = "/data/ira_sft_v1/checkpoint-22191"
    dpo_path    = "/data/ira_dpo_v2/dpo_checkpoint"
    export_path = "/data/ira_final_sglang_16bit"

    # ── Step 1: Load SFT checkpoint via Unsloth ───────────────────────────────
    # FastLanguageModel.from_pretrained understands Unsloth's own adapter format
    # (including Gemma4ClippableLinear) — PeftModel.from_pretrained does NOT.
    print("Loading SFT checkpoint via Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=sft_path,
        max_seq_length=4096,
        load_in_4bit=False,
        dtype=torch.bfloat16,
    )
    model = model.merge_and_unload()
    print("SFT adapters merged.")

    # ── Step 2: Load DPO checkpoint via Unsloth ───────────────────────────────
    print("Loading DPO checkpoint via Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=dpo_path,
        max_seq_length=4096,
        load_in_4bit=False,
        dtype=torch.bfloat16,
    )
    model = model.merge_and_unload()
    print("DPO adapters merged.")

    # ── Step 3: Save fully merged 16-bit model ────────────────────────────────
    # SGLang loads directly from a standard HF checkpoint directory —
    # no special export step required beyond save_pretrained.
    print(f"Saving fully merged 16-bit model for SGLang to {export_path}...")
    model.save_pretrained(export_path)
    tokenizer.save_pretrained(export_path)
    volume.commit()
    print(f"Model successfully saved to {export_path}")

@app.local_entrypoint()
def main():
    export_model.remote()
