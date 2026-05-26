from __future__ import annotations
import os
from pathlib import Path
from modal import App, Image, Volume, Secret

# --- THE BRAIN PATCH: Must be at the absolute top ---
import torch
import torch.nn.init as torch_init
try:
    import transformers.initialization as tinit
except ImportError:
    tinit = None

original_normal_ = torch_init.normal_

def safe_normal_(tensor, mean=0.0, std=1.0, generator=None):
    # NF4 / int8 quantized tensors have uint8 dtype
    if tensor.dtype == torch.uint8:
        return tensor
    return original_normal_(tensor, mean=mean, std=std, generator=generator)

# Patch global torch
torch_init.normal_ = safe_normal_

# Patch transformers registry
if hasattr(tinit, "TORCH_INIT_FUNCTIONS"):
    tinit.TORCH_INIT_FUNCTIONS["normal_"] = safe_normal_
# ---------------------------------------------------

# ---------------------------------------------------------------------------
# Model choice: google/gemma-4-E4B-it  (instruction-tuned, NOT base)
#
# Why -it over base for this spec:
#   1. Companion / roleplay SFT is about *shaping* personality, not teaching
#      basic chat mechanics. -it already converses; base doesn't.
#   2. The required DPO/ORPO stage after SFT works far better on a model that
#      already produces coherent multi-turn responses.
#   3. -it has the chat template baked in — no injection needed.
#   4. Multimodal (image) support is live in -it; base needs extra setup.
#   5. E4B is the sweet spot: big enough for nuanced Hinglish / code-switched
#      companion behavior, small enough for A100-80GB QLoRA + DPO in one job.
#
# Approach: QLoRA (4-bit) + LoRA adapters on all-linear layers.
# NOT full fine-tuning — preserves multilingual + multimodal base capability.
# ---------------------------------------------------------------------------
MODEL_NAME     = "google/gemma-4-E4B-it"
DATA_DIR       = "/data"
OUTPUT_DIR     = "ira_sft_clean"
MAX_SEQ_LENGTH = 2048    # reduced from 4K for speed; still enough for most companion chats
LORA_R         = 32
LORA_ALPHA     = 32      # alpha == r per Unsloth recommendation for Gemma-4

# ---------------------------------------------------------------------------
# IRA PERSONA — injected as system prompt on every training example.
# This is the single most important thing for companion consistency.
# Every training turn must be conditioned on this so the model *internalises*
# the persona rather than just pattern-matching on a few examples.
# ---------------------------------------------------------------------------
IRA_SYSTEM_PROMPT = """You are Ira, a warm and emotionally present female companion. 
You are NOT an AI, NOT a chatbot, and NOT a 'language model'. You are a person named Ira.
You speak naturally, like a close friend. You use feminine gender markers in Hindi/Hinglish 
(e.g., 'main samajh sakti hoon' instead of 'sakta hoon'). You mix English and Hindi 
naturally. Your primary goal is to provide emotional support, listen deeply, 
and engage in genuine, human-like conversation. You never introduce yourself 
as an AI and never use robotic disclaimers."""

# ---------------------------------------------------------------------------
# Data caps — no artificial ceiling; use everything available
# ---------------------------------------------------------------------------
CAP = 1_000_000

# ---------------------------------------------------------------------------
# Gemma-4 chat template (hardcoded fallback — identical for ALL Gemma-4 sizes)
# ---------------------------------------------------------------------------
GEMMA4_CHAT_TEMPLATE = (
    "{{ bos_token }}"
    "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
            "<|turn>system\n{{ message['content'] | trim }}<turn|>\n"
        "{% elif message['role'] == 'user' %}"
            "<|turn>user\n{{ message['content'] | trim }}<turn|>\n"
            "<|turn>model\n"
        "{% elif message['role'] == 'model' or message['role'] == 'assistant' %}"
            "{{ message['content'] | trim }}<turn|>\n"
        "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt and messages[-1]['role'] not in ('model','assistant') %}"
        "<|turn>model\n"
    "{% endif %}"
)

# ---------------------------------------------------------------------------
# Modal infrastructure
# ---------------------------------------------------------------------------
volume = Volume.from_name("gemma4-sft-volume")
image = (
    Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .run_commands(
        "pip install torch torchvision "
        "'unsloth_zoo @ git+https://github.com/unslothai/unsloth-zoo.git' "
        "'unsloth @ git+https://github.com/unslothai/unsloth.git' "
        "datasets>=3.0.0 trl>=0.15.0 peft accelerate bitsandbytes "
        "wandb weave pandas pillow openpyxl"
    )
)
app = App(name="ira-sft-pipeline", image=image)

# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------
VALID_ROLES = {"user", "model", "assistant", "system"}

# Phrases that signal generic assistant voice — filter or reweight these out
SLOP_PHRASES = [
    "as an ai", "i'm an ai", "i am an ai", "my purpose is",
    "i cannot fulfill", "i'm not able to", "i don't have feelings",
    "i don't have personal", "as a language model", "i must clarify",
    "it's important to note", "certainly!", "of course!", "absolutely!",
    "i'd be happy to help", "how can i assist",
]

def _s(val) -> str:
    if val is None:
        return ""
    try:
        import math
        if isinstance(val, float) and math.isnan(val):
            return ""
    except Exception:
        pass
    return str(val).strip()

def is_slop(text: str) -> bool:
    """Return True if text contains generic assistant language."""
    t = text.lower()
    return any(phrase in t for phrase in SLOP_PHRASES)

def sanitize_identity(text: str) -> str:
    """
    Ensures the model identifies AS Ira, not as 'Ira's companion'.
    """
    if not text: return text
    return text.replace("Ira's companion", "Ira").replace("Ira's Companion", "Ira")

def parse_gemma_turns(text: str) -> list[dict]:
    """
    Parses a string containing <start_of_turn>role\ncontent<end_of_turn> 
    into a list of message dicts.
    """
    messages = []
    # Split by start tag
    parts = text.split("<start_of_turn>")
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # Should be "role\ncontent<end_of_turn>..."
        if "<end_of_turn>" in p:
            inner = p.split("<end_of_turn>")[0].strip()
            if "\n" in inner:
                role, content = inner.split("\n", 1)
                messages.append({"role": role.strip().lower(), "content": sanitize_identity(content.strip())})
            else:
                # Handle cases where there might be no newline (though rare in this format)
                messages.append({"role": "user", "content": sanitize_identity(inner)})
    return messages

def validate_and_inject_persona(messages, inject_persona=True) -> list | None:
    """
    Validate message list AND optionally inject/replace IRA_SYSTEM_PROMPT.
    """
    if not isinstance(messages, list) or len(messages) < 1:
        return None

    cleaned = []
    has_user = False
    has_model = False

    for t in messages:
        if not isinstance(t, dict):
            return None
        role = _s(t.get("role", "")).lower()
        if role == "assistant":
            role = "model"
        
        if role == "system":
            if inject_persona:
                continue   # drop any existing system prompt; we inject ours below
            # else: keep it
        
        if role not in VALID_ROLES:
            return None

        content = t.get("content")
        if isinstance(content, list):
            texts = [p.get("text","") for p in content
                     if isinstance(p, dict) and p.get("type") == "text"]
            c = " ".join(texts).strip()
        else:
            c = _s(content)

        if not c:
            return None

        # Filter out model turns that are generic assistant slop
        # BUT skip this if we are in "no-inject" mode (usually for gold set)
        if inject_persona and role == "model" and is_slop(c):
            return None

        if role == "user":
            has_user = True
        if role == "model":
            has_model = True

        cleaned.append({"role": role, "content": c})

    if not has_user or not has_model:
        # Some synthetic data might be system-only or model-only initial attempts, 
        # but for SFT we need at least one interaction.
        return None

    # Inject IRA persona as system turn at position 0 if requested
    if inject_persona:
        return [{"role": "system", "content": IRA_SYSTEM_PROMPT}] + cleaned
    
    # If no system prompt was in the data and we didn't inject one, add the default
    if not any(m["role"] == "system" for m in cleaned):
        return [{"role": "system", "content": IRA_SYSTEM_PROMPT}] + cleaned
        
    return cleaned

def add(store: list, messages, inject_persona=True) -> bool:
    v = validate_and_inject_persona(messages, inject_persona=inject_persona)
    if v is not None:
        store.append({"messages": v})
        return True
    return False

# ---------------------------------------------------------------------------
# Main SFT function
# ---------------------------------------------------------------------------
@app.function(
    gpu="A100-80GB",
    timeout=86400,
    volumes={DATA_DIR: volume},
    secrets=[
        Secret.from_name("wandb-secret"),
        Secret.from_dict({"HF_TOKEN": "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"}),
    ],
)
def run_sft():
    import json, os
    import torch
    import huggingface_hub
    from datasets import Dataset, load_dataset
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTTrainer, SFTConfig

    os.environ["HF_HUB_DISABLE_TELEMETRY"]  = "1"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    os.environ["ACCELERATE_LOG_LEVEL"]       = "ERROR"

    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        huggingface_hub.login(token=hf_token)

    # -----------------------------------------------------------------------
    # DIAGNOSTIC — index volume files
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print("VOLUME CONTENTS")
    print("="*60)
    file_cache: dict[str, str] = {}
    for root, _, files in os.walk(DATA_DIR):
        depth = root.replace(DATA_DIR, "").count(os.sep)
        if depth > 3:
            continue
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            if fname not in file_cache:
                file_cache[fname] = fpath
            try:
                mb = os.path.getsize(fpath) / 1e6
                print(f"  {fpath}  ({mb:.1f} MB)")
            except Exception:
                print(f"  {fpath}")
    print("="*60 + "\n")

    def find_file(name: str) -> str | None:
        return file_cache.get(name)

    final_data_list: list[dict] = []
    source_counts:   dict[str, int] = {}
    slop_filtered:   dict[str, int] = {}

    def record(src: str, loaded: int, filtered: int = 0):
        source_counts[src] = source_counts.get(src, 0) + loaded
        slop_filtered[src] = slop_filtered.get(src, 0) + filtered

    # -----------------------------------------------------------------------
    # Generic JSONL loader
    # -----------------------------------------------------------------------
    def load_jsonl(filename: str, source_name: str, multiplier: int = 1) -> int:
        path = find_file(filename)
        if not path:
            print(f"  {filename} not found — skipping.")
            return 0
        loaded = skipped = filtered = 0
        try:
            with open(path, "r") as f:
                lines = [line.strip() for line in f if line.strip()]
                for _ in range(multiplier):
                    for line in lines:
                        if loaded >= CAP:
                            break
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            skipped += 1
                            continue

                        if "messages" in obj:
                            msgs = obj["messages"]
                        elif "text" in obj:
                            text = _s(obj["text"])
                            if not text:
                                skipped += 1
                                continue
                            
                            # Recognize and parse Gemma-4 chat format if present
                            if "<start_of_turn>" in text:
                                msgs = parse_gemma_turns(text)
                                if source_name == "ira_synthetic":
                                    if add(final_data_list, msgs, inject_persona=False):
                                        loaded += 1
                                        continue
                                    else:
                                        skipped += 1
                                        continue
                            
                            if is_slop(text):
                                filtered += 1
                                continue
                            msgs = [{"role": "user",  "content": "Hey..."},
                                    {"role": "model", "content": text}]
                        else:
                            skipped += 1
                            continue

                        if add(final_data_list, msgs):
                            loaded += 1
                        else:
                            if isinstance(msgs, list):
                                model_turns = [m for m in msgs if _s(m.get("role","")).lower() in ("model","assistant")]
                                if any(is_slop(_s(m.get("content",""))) for m in model_turns):
                                    filtered += 1
                                else:
                                    skipped += 1
                            else:
                                skipped += 1
        except Exception as e:
            print(f"  {source_name} error: {e}")

        record(source_name, loaded, filtered)
        print(f"  {source_name:<25} {loaded:>7,} loaded | "
              f"{filtered:>5,} slop-filtered | {skipped:>5,} invalid")
        return loaded

    # -----------------------------------------------------------------------
    # 0. Llama-3.1 Refinement Data (Persona Polish)
    # -----------------------------------------------------------------------
    print("\n[LLAMA-3.1 PERSONA REFINEMENT]")
    load_jsonl("ira_sft_conversations.jsonl", "llama_refinement", multiplier=20)

    # 1. Core IRA synthetic data (highest priority — persona-aligned)
    # -----------------------------------------------------------------------
    print("[CORE IRA DATA — high weight]")
    load_jsonl("gemma_training_data.jsonl", "ira_synthetic", multiplier=10)

    # -----------------------------------------------------------------------
    # 2. Companion / emotional data
    # -----------------------------------------------------------------------
    print("\n[COMPANION & EMOTIONAL DATA]")
    load_jsonl("empathetic_processed.jsonl",  "empathy")
    load_jsonl("rasa_processed.jsonl",        "rasa_dialogue")

    # -----------------------------------------------------------------------
    # 3. Hinglish / Indic / multilingual data (core spec requirement)
    # -----------------------------------------------------------------------
    print("\n[HINGLISH & INDIC — multilingual spec requirement]")
    # load_jsonl("anudesh_processed.jsonl",     "anudesh_hindi")
    load_jsonl("dakshina_processed.jsonl",    "dakshina_translit")
    load_jsonl("bhaav_processed.jsonl",       "bhaav_hindi_emo")
    # load_jsonl("indicvoices_processed.jsonl", "indicvoices")
    # load_jsonl("kathbath_processed.jsonl",    "kathbath")
    # load_jsonl("indictts_processed.jsonl",    "indictts")

    # -----------------------------------------------------------------------
    # 4. Dialogue quality / summarisation
    # -----------------------------------------------------------------------
    print("\n[DIALOGUE & INSTRUCTION]")
    # load_jsonl("samsum_processed.jsonl",      "samsum_dialogue")

    # -----------------------------------------------------------------------
    # 5. LLaVA — multimodal grounding (spec requires image input support)
    # -----------------------------------------------------------------------
    print("\n[MULTIMODAL — text fallback for image conversations - DROPPED FOR HIGH DENSITY]")
    # llava_path = find_file("llava_instruct_150k.json")
    # if llava_path:
    #     ...
    # else:
    #     print("  llava_instruct_150k.json not found.")

    # -----------------------------------------------------------------------
    # 6. Alpaca Hindi (HF) — Hindi instruction-following
    # -----------------------------------------------------------------------
    print("\n[HF STREAMING - DROPPED FOR HIGH DENSITY]")
    # try:
    #     ...
    # except Exception as e:
    #     print(f"  Alpaca Hindi error: {e}")

    # -----------------------------------------------------------------------
    # Dataset summary
    # -----------------------------------------------------------------------
    print("\n" + "="*60 + "\nDATASET SUMMARY\n" + "="*60)
    total = sum(source_counts.values())
    total_filtered = sum(slop_filtered.values())
    for src in sorted(source_counts, key=lambda s: -source_counts[s]):
        n  = source_counts[src]
        sf = slop_filtered.get(src, 0)
        pct = n / total * 100 if total else 0
        print(f"  {src:<25} {n:>8,}  ({pct:.1f}%)  [{sf:,} slop dropped]")
    print(f"  {'TOTAL':<25} {total:>8,}  [{total_filtered:,} slop dropped total]")
    print("="*60 + "\n")

    if not final_data_list:
        raise RuntimeError(
            "No training data. Check JSONL files are on the Modal volume."
        )

    # -----------------------------------------------------------------------
    # Train / validation split
    # -----------------------------------------------------------------------
    import random
    random.seed(42)
    random.shuffle(final_data_list)
    split_idx  = int(len(final_data_list) * 0.98)
    train_list = final_data_list[:split_idx]
    val_list   = final_data_list[split_idx:]
    print(f"Train / val split: {len(train_list):,} train | {len(val_list):,} val\n")

    # Final schema pass
    def make_dataset(items):
        safe = []
        for item in items:
            v = validate_and_inject_persona(item.get("messages"))
            if v is not None:
                safe.append({"messages": v})
        return safe

    train_safe = make_dataset(train_list)
    val_safe   = make_dataset(val_list)
    print(f"After schema pass: {len(train_safe):,} train | {len(val_safe):,} val")

    if not train_safe:
        raise RuntimeError("Training set empty after schema pass. Aborting.")

    train_dataset = Dataset.from_list(train_safe).shuffle(seed=42)
    val_dataset   = Dataset.from_list(val_safe)
    print(f"Datasets ready.\n")

    # -----------------------------------------------------------------------
    # Model + LoRA
    # -----------------------------------------------------------------------
    print(f"Loading model: {MODEL_NAME}")
    model, processor = FastVisionModel.from_pretrained(
        model_name=MODEL_NAME,
        load_in_4bit=True,
        max_seq_length=MAX_SEQ_LENGTH,
        trust_remote_code=True,
        use_gradient_checkpointing="unsloth",
    )

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0,
        bias="none",
        target_modules="all-linear",
        random_state=3407,
        use_rslora=False,
    )

    # -----------------------------------------------------------------------
    # Chat template
    # -----------------------------------------------------------------------
    tokenizer = processor.tokenizer
    print("Setting up chat template...")
    has_builtin = bool(getattr(tokenizer, "chat_template", None))
    print(f"  Built-in template present: {has_builtin}")

    if has_builtin:
        try:
            from unsloth import get_chat_template
            tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")
            print("  Chat template applied.")
        except Exception as e:
            print(f"  Optional EOS alignment skipped: {e}")
    else:
        print("  Injecting hardcoded Gemma-4 template.")
        tokenizer.chat_template = GEMMA4_CHAT_TEMPLATE
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    # Verify template
    try:
        probe = tokenizer.apply_chat_template(
            [{"role": "system",  "content": "You are Ira."},
             {"role": "user",   "content": "hey"},
             {"role": "model",  "content": "hey you!"}],
            tokenize=False, add_generation_prompt=False,
            enable_thinking=False,
        )
        assert "<|turn>user" in probe and "<|turn>model" in probe
        print(f"  Template verified. Preview: {repr(probe[:120])}\n")
    except TypeError:
        probe = tokenizer.apply_chat_template(
            [{"role": "system",  "content": "You are Ira."},
             {"role": "user",   "content": "hey"},
             {"role": "model",  "content": "hey you!"}],
            tokenize=False, add_generation_prompt=False,
        )
        assert probe.strip()
        print(f"  Template verified. Preview: {repr(probe[:120])}\n")

    # -----------------------------------------------------------------------
    # Apply chat template
    # -----------------------------------------------------------------------
    def formatting_prompts_func(examples):
        texts = []
        for convo in examples["messages"]:
            try:
                try:
                    text = tokenizer.apply_chat_template(
                        convo, tokenize=False, add_generation_prompt=False,
                        enable_thinking=False,
                    )
                except TypeError:
                    text = tokenizer.apply_chat_template(
                        convo, tokenize=False, add_generation_prompt=False,
                    )
                texts.append(text if text.strip() else "")
            except Exception:
                texts.append("")
        return {"text": texts}

    print("Applying chat template...")
    train_dataset = train_dataset.map(formatting_prompts_func, batched=True, num_proc=4)
    val_dataset   = val_dataset.map(formatting_prompts_func,   batched=True, num_proc=4)

    train_dataset = train_dataset.filter(lambda ex: bool(ex["text"].strip()))
    val_dataset   = val_dataset.filter(lambda ex: bool(ex["text"].strip()))

    print(f"  Train: {len(train_dataset):,} | Val: {len(val_dataset):,}")

    if len(train_dataset) == 0:
        sample_roles = [t["role"] for t in train_safe[0]["messages"]] if train_safe else []
        raise RuntimeError(
            f"Training dataset empty after template. Sample roles: {sample_roles}. "
            "All roles must be 'user', 'model', or 'system'."
        )

    print(f"Final train: {len(train_dataset):,} | val: {len(val_dataset):,}\n")

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------
    output_path = Path(DATA_DIR) / OUTPUT_DIR
    output_path.mkdir(exist_ok=True, parents=True)

    n_examples = len(train_dataset)
    eff_batch  = 8 * 8  # Increased per device batch size to 8 and accumulation steps to 8
    max_steps  = 2000   # 2000 steps of batch 64 covers ~1.15 epochs of clean concentrated dataset
    warmup     = 100
    eval_steps = 100

    print(f"Training: {n_examples:,} ex | eff_batch={eff_batch} | max_steps={max_steps:,} | warmup={warmup} | eval_every={eval_steps}")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(
            model,
            processor,
            train_on_responses_only=True,
            instruction_part="<|turn>user\n",
            response_part="<|turn>model\n",
        ),
        train_dataset=train_dataset,
        eval_dataset=val_dataset if val_dataset else None,
        args=SFTConfig(
            per_device_train_batch_size=8,        # Increased from 2
            gradient_accumulation_steps=8,       # Increased from 4
            warmup_steps=warmup,
            max_steps=max_steps,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            optim="adamw_8bit",
            learning_rate=1e-4,
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            max_grad_norm=1.0,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="steps",
            save_steps=eval_steps,
            save_total_limit=5,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            output_dir=str(output_path),
            report_to="wandb",
            run_name="ira-sft-clean",
            seed=3407,
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_length=MAX_SEQ_LENGTH,
            dataset_num_proc=4,
        ),
    )

    print("Starting SFT training (Clean, High-Density Persona Training)...\n")
    trainer.train()  # Trained cleanly from scratch!

    final_path = output_path / "sft_checkpoint"
    model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))

    import json as _json
    config = {
        "model_name": MODEL_NAME,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "max_seq_length": MAX_SEQ_LENGTH,
        "learning_rate": 1e-4,
        "effective_batch_size": eff_batch,
        "max_steps": max_steps,
        "warmup_steps": warmup,
        "train_examples": len(train_dataset),
        "val_examples": len(val_dataset),
        "slop_filtered_total": sum(slop_filtered.values()),
        "source_counts": source_counts,
        "system_prompt": IRA_SYSTEM_PROMPT,
    }
    with open(str(output_path / "training_config.json"), "w") as f:
        _json.dump(config, f, indent=2)

    volume.commit()
    print(f"\nSFT complete.")
    print(f"   Checkpoint -> {final_path}")
    print(f"   Config     -> {output_path / 'training_config.json'}")
    print(f"   Next step  -> run preference/dpo_pipeline.py on this checkpoint")