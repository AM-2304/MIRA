from modal import App, Image, Secret, Volume
from pathlib import Path

# --- CONFIGURATION ---
MODEL_NAME = "google/gemma-4-E4B-it"
MAX_SEQ_LENGTH = 4096
DATA_FILE = "gemma_training_data.jsonl"

VOLUME_NAME = "gemma4-sft-volume"
DATA_DIR = "/data"
OUTPUT_DIR = "results"

NUM_EPOCHS = 3
LEARNING_RATE = 2e-4
PER_DEVICE_BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 8
LORA_R = 64
LORA_ALPHA = 128

volume = Volume.from_name(VOLUME_NAME, create_if_missing=True)

# KEY FIX: Install everything in ONE pip call, NO extra_index_url.
# Modal's internal mirror has torch 2.11+ which satisfies:
#   - torchao 0.17.0 (needs torch.int1, _pytree.register_constant)
#   - unsloth_zoo 2026.4.9
# Letting pip resolve all constraints at once avoids version conflicts.
image = (
    Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .run_commands(
        "pip install torch torchvision "
        "'unsloth @ git+https://github.com/unslothai/unsloth.git' "
        "'unsloth_zoo @ git+https://github.com/unslothai/unsloth-zoo.git' "
        "'datasets>=3.4.1,<4.4.0' "   # unsloth_zoo requires datasets<4.4.0
        "'trl>=0.18.2,<=0.24.0' "     # unsloth_zoo requires trl in this range
        "peft accelerate bitsandbytes sentencepiece hf_transfer wandb weave"
    )
    .env({
        "HF_HOME": "/model_cache",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    })
)

app = App(name="gemma4-e4b-sft-final", image=image)

@app.function(
    gpu="A100-80GB",  # Production GPU
    timeout=19800,    # 5.5 hours 
    volumes={DATA_DIR: volume},
    secrets=[Secret.from_name("wandb-secret")],
    max_containers=1,
)
def run_sft_training():
    import os
    import weave
    import torch
    from pathlib import Path
    from datasets import load_dataset
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments

    if os.environ.get("WANDB_API_KEY"):
        weave.init("gemma4-ira-companion")

    print(f"torch={torch.__version__} | GPU={torch.cuda.get_device_name(0)}")
    print(f"Starting SFT: {MODEL_NAME}")
    print(f"Effective Batch Size: {PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION}")

    # Load base model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )

    # Apply LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # Load training data
    data_path = Path(DATA_DIR) / DATA_FILE
    try:
        dataset = load_dataset("json", data_files=str(data_path), split="train")
        print(f"Loaded {len(dataset)} training examples")
    except Exception as e:
        print(f"ERROR loading dataset: {e}")
        return

    output_checkpoint_path = Path(DATA_DIR) / OUTPUT_DIR
    output_checkpoint_path.mkdir(exist_ok=True, parents=True)
    report_to_wandb = "wandb" if os.environ.get("WANDB_API_KEY") else "none"

    training_args = TrainingArguments(
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        bf16=True,
        logging_steps=10,
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
        seed=42,
        output_dir=str(output_checkpoint_path),
        save_strategy="epoch",
        report_to=report_to_wandb,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=2,
        packing=False, # Disabled for maximum quality; one sample per sequence
        args=training_args,
    )

    print("\nTraining...")
    trainer.train()
    print("Fine-Tuning Complete!")

    final_save_path = output_checkpoint_path / "final_checkpoint"
    final_save_path.mkdir(exist_ok=True)
    trainer.model.save_pretrained(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    volume.commit()

    print(f"\nCheckpoint saved to {final_save_path}")

if __name__ == "__main__":
    run_sft_training.remote()