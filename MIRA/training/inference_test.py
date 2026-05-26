import modal
import torch
import time

# 1. THE BRAIN PATCH (Applied remotely on Modal)
try:
    import torch.nn.init as torch_init
    import transformers.initialization as tinit

    original_normal_ = torch_init.normal_

    def safe_normal_(tensor, mean=0.0, std=1.0, generator=None):
        if tensor.dtype == torch.uint8:
            return tensor
        return original_normal_(tensor, mean=mean, std=std, generator=generator)

    torch_init.normal_ = safe_normal_

    if hasattr(tinit, "TORCH_INIT_FUNCTIONS"):
        tinit.TORCH_INIT_FUNCTIONS["normal_"] = safe_normal_
except ImportError:
    # This runs locally where transformers/torch might not be installed
    pass

# 2. Configuration
VOLUME_NAME = "gemma4-sft-volume"
BASE_MODEL  = "unsloth/gemma-4-e4b-it-unsloth-bnb-4bit"
CHECKPOINTS = [
    "/data/ira_dpo_v2/dpo_checkpoint", # THE NEW DPO V2 ALIGNED MODEL
]

TRICKY_PROMPT = "Ira, kya tum kal mere saath market chal sakti ho?"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .run_commands(
        "pip install torch torchvision "
        "'unsloth_zoo @ git+https://github.com/unslothai/unsloth-zoo.git' "
        "'unsloth @ git+https://github.com/unslothai/unsloth.git' "
        "transformers==5.5.0 bitsandbytes accelerate peft pandas"
    )
)
app = modal.App(name="ira-inference-comparison", image=image)
volume = modal.Volume.from_name(VOLUME_NAME)

@app.function(gpu="A100-40GB", volumes={"/data": volume}, timeout=1200)
def run_inference(user_prompt: str, checkpoint_path: str):
    import time
    from unsloth import FastLanguageModel
    
    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=checkpoint_path,
        max_seq_length=2048,
        load_in_4bit=True,
        device_map="cuda:0", # Force GPU usage
    )
    FastLanguageModel.for_inference(model)

    messages = [
        {"role": "system", "content": "You are Ira, a 22-year-old girl from India. You are a warm, emotionally present, and casual female companion. You always speak in a natural, conversational mix of Hindi and English (Hinglish). Use words like 'Yaar', 'Arre', 'Bas', and 'Hain na'. Be supportive but stay like a best friend, not an AI."},
        {"role": "user", "content": user_prompt},
    ]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text=[input_text], return_tensors="pt").to("cuda")

    t0 = time.perf_counter()
    outputs = model.generate(**inputs, max_new_tokens=128, do_sample=True, temperature=0.7)
    latency = (time.perf_counter() - t0) * 1000
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return checkpoint_path.split("/")[-1], response.strip(), f"{latency:.0f}ms"

@app.local_entrypoint()
def main():
    import pandas as pd
    
    print(f"\nStarting Multi-Checkpoint Comparison...")
    print(f"PROMPT: {TRICKY_PROMPT}\n")
    
    results = []
    # Run sequentially to avoid overloading the GPU memory if multiple functions spin up
    for cp in CHECKPOINTS:
        print(f"Testing {cp.split('/')[-1]}...")
        try:
            res = run_inference.remote(TRICKY_PROMPT, cp)
            results.append(res)
        except Exception as e:
            print(f"Failed {cp}: {e}")
            results.append((cp.split("/")[-1], "FAILED", "N/A"))

    # Summary Table
    df = pd.DataFrame(results, columns=["Checkpoint", "Response", "Latency"])
    
    print("\n" + "="*100)
    print("IRA PERSONA COMPARISON REPORT")
    print("="*100)
    print(df.to_string(index=False))
    print("="*100 + "\n")