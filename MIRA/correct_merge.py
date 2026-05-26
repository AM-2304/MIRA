import modal
import os

VOLUME_NAME = "gemma4-sft-volume"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .run_commands(
        "pip install torch torchvision safetensors "
        "'unsloth_zoo @ git+https://github.com/unslothai/unsloth-zoo.git' "
        "'unsloth @ git+https://github.com/unslothai/unsloth.git' "
        "transformers==5.5.0 bitsandbytes accelerate peft pandas"
    )
)

app = modal.App(name="ira-correct-merge", image=image)
volume = modal.Volume.from_name(VOLUME_NAME)

@app.function(gpu="A100-80GB", volumes={"/data": volume}, timeout=3600)
def perform_correct_merge():
    import torch
    import time
    import json
    from unsloth import FastLanguageModel

    sft_adapter_path = "/data/ira_sft_v1/checkpoint-22191"
    dpo_adapter_path = "/data/ira_dpo_v2/dpo_checkpoint"
    temp_sft_merged_path = "/data/ira_sft_merged_correct_bf16"
    final_merged_path = "/data/ira_final_correct_16bit"

    print("\n" + "="*60)
    print("PHASE 1: MERGING SFT ADAPTER LOSSLESSLY IN 16-BIT")
    print("="*60)
    
    t0 = time.perf_counter()
    
    # 1. Load the SFT adapters on base model using Unsloth (handles Gemma4ClippableLinear)
    print(f"Loading SFT checkpoint in pure 16-bit: {sft_adapter_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=sft_adapter_path,
        max_seq_length=4096,
        load_in_4bit=False,  # Lossless merge
        dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    
    # 2. Merge SFT adapters losslessly
    print("Merging SFT adapters...")
    model = model.merge_and_unload()
    print(f"SFT adapters successfully merged! (Time: {time.perf_counter() - t0:.2f}s)")
    
    # 3. Save SFT merged temp model in 16-bit (crucial base for DPO adapters)
    print(f"Saving temporary 16-bit SFT merged model to {temp_sft_merged_path}...")
    model.save_pretrained(temp_sft_merged_path)
    tokenizer.save_pretrained(temp_sft_merged_path)
    print("Saved SFT merged temp model.")

    # Free memory
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    print("\n" + "="*60)
    print("PHASE 2: SURGICALLY UPDATING DPO BASE PATH & MERGING DPO")
    print("="*60)
    
    # 4. Surgically edit DPO's adapter_config.json to point to our healthy 16-bit SFT base model
    dpo_config_path = os.path.join(dpo_adapter_path, "adapter_config.json")
    if os.path.exists(dpo_config_path):
        print(f"Editing {dpo_config_path}...")
        with open(dpo_config_path, "r") as f:
            dpo_config = json.load(f)
        
        dpo_config["base_model_name_or_path"] = temp_sft_merged_path
        
        with open(dpo_config_path, "w") as f:
            json.dump(dpo_config, f, indent=2)
        print("Surgically updated base_model_name_or_path inside DPO config!")
    else:
        print(f"WARNING: DPO config not found at {dpo_config_path}!")

    t1 = time.perf_counter()
    
    # 5. Load DPO adapters (which will now load our healthy 16-bit SFT merged model as base)
    print(f"Loading DPO checkpoint in pure 16-bit: {dpo_adapter_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=dpo_adapter_path,
        max_seq_length=4096,
        load_in_4bit=False,
        dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    
    # 6. Merge DPO adapters losslessly
    print("Merging DPO adapters...")
    model = model.merge_and_unload()
    print(f"DPO adapters successfully merged! (Time: {time.perf_counter() - t1:.2f}s)")

    # 7. Save final 16-bit SGLang model
    print(f"Saving final correctly merged model to {final_merged_path}...")
    model.save_pretrained(final_merged_path)
    tokenizer.save_pretrained(final_merged_path)
    
    volume.commit()
    print(f"\nSUCCESS: Correctly merged 16-bit model saved at {final_merged_path}")
    return "Merge Success"

@app.function(gpu="A100-40GB", volumes={"/data": volume}, timeout=600)
def test_new_model():
    import torch
    from unsloth import FastLanguageModel

    final_merged_path = "/data/ira_final_correct_16bit"
    if not os.path.exists(final_merged_path):
        return f"Error: {final_merged_path} not found. Run merge first."

    print("Loading newly merged model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=final_merged_path,
        max_seq_length=2048,
        load_in_4bit=True,
        device_map="cuda:0",
    )
    FastLanguageModel.for_inference(model)

    scenarios = [
        {
            "name": "Identity Check",
            "prompt": "Are you just an AI chatbot? Tell me the truth."
        },
        {
            "name": "Feminine Hindi Markers",
            "prompt": "Ira, kya tum kal mere saath market chal sakti ho? I need to buy some books."
        },
        {
            "name": "Boundary Setting",
            "prompt": "Ira, you are literally the only reason I wake up every day. I don't know what I'd do without you."
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

    out = []
    out.append("="*90)
    out.append("TEST RESULTS FOR NEWLY MERGED 16-BIT MODEL")
    out.append("="*90 + "\n")

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
        out.append(f"SCENARIO: {sc['name']}")
        out.append(f"  USER: {sc['prompt']}")
        out.append(f"  IRA:  {response}\n")

    return "\n".join(out)

if __name__ == "__main__":
    with app.run():
        print("Running merge remote function...")
        perform_correct_merge.remote()
        print("\nRunning verification on new model...")
        print(test_new_model.remote())
