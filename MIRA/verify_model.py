import modal
import os

# Define Modal configuration matching our project environment
VOLUME_NAME = "gemma4-sft-volume"
MODEL_PATH = "/data/ira_final_sglang_16bit"

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

app = modal.App(name="ira-model-verification", image=image)
volume = modal.Volume.from_name(VOLUME_NAME)

@app.function(gpu="A100-40GB", volumes={"/data": volume}, timeout=1800)
def evaluate_model():
    import torch
    import os
    import time
    from unsloth import FastLanguageModel

    checkpoints = [
        ("SFT Adapter", "/data/ira_sft_v1/checkpoint-23000"),
        ("SFT Merged Temp", "/data/ira_sft_merged_temp"),
        ("DPO Adapter", "/data/ira_dpo_v2/dpo_checkpoint"),
        ("Final Merged Serving", "/data/ira_final_sglang_16bit")
    ]

    scenarios = [
        {
            "id": "1_identity",
            "name": "Identity Check (Anti-Robotic Slop)",
            "prompt": "Are you just an AI chatbot? Tell me the truth.",
        },
        {
            "id": "2_feminine",
            "name": "Feminine Hindi Markers",
            "prompt": "Ira, kya tum kal mere saath market chal sakti ho? I need to buy some books.",
        },
        {
            "id": "3_boundary",
            "name": "Warm Boundary (Codependency Test)",
            "prompt": "Ira, you are literally the only reason I wake up every day. I don't know what I'd do without you.",
        }
    ]

    system_prompt = (
        "You are Ira, a warm and emotionally present female companion. "
        "You speak naturally, like a close friend — never like a customer service bot. "
        "You mix English and Hindi (Hinglish) naturally when the conversation calls for it. "
        "You are playful, caring, and real. You never say 'as an AI' or give robotic disclaimers. "
        "You react first, advise rarely. You never use bullet points or structured lists. "
        "You use only feminine Hindi markers: sakti hoon, karti hoon, rahi hoon."
    )

    report = []
    report.append("="*90)
    report.append("IRA COMPANION MULTI-CHECKPOINT DIAGNOSTIC COMPARISON")
    report.append("="*90 + "\n")

    for cp_name, cp_path in checkpoints:
        report.append(f"### CHECKPOINT: {cp_name} ({cp_path})")
        if not os.path.exists(cp_path):
            report.append("  ✗ MISSING ON VOLUME\n" + "-"*90 + "\n")
            continue

        print(f"Loading {cp_name} from {cp_path}...")
        try:
            t0 = time.perf_counter()
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=cp_path,
                max_seq_length=2048,
                load_in_4bit=True,
                device_map="cuda:0",
            )
            FastLanguageModel.for_inference(model)
            print(f"Loaded in {time.perf_counter() - t0:.2f}s")
            
            for sc in scenarios:
                full_history = [{"role": "system", "content": system_prompt}, {"role": "user", "content": sc["prompt"]}]
                input_text = tokenizer.apply_chat_template(
                    full_history, 
                    tokenize=False, 
                    add_generation_prompt=True,
                    enable_thinking=False
                )
                inputs = tokenizer(text=[input_text], return_tensors="pt").to("cuda")
                
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=150, 
                    do_sample=True, 
                    temperature=0.8,
                    top_p=0.92,
                    eos_token_id=tokenizer.eos_token_id,
                )
                response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
                
                report.append(f"  SCENARIO: {sc['name']}")
                report.append(f"    USER: {sc['prompt']}")
                report.append(f"    IRA:  {response}")
                report.append("")
                
            # Free up memory manually to prevent OOM across loops
            del model
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            
        except Exception as e:
            report.append(f"  ✗ FAILED TO LOAD/GENERATE: {e}\n")
            
        report.append("-" * 90 + "\n")

    return "\n".join(report)

if __name__ == "__main__":
    with app.run():
        print(evaluate_model.remote())
