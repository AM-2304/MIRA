import modal
import os

# Use the same image setup as the main project
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .run_commands(
        "pip install torch torchvision safetensors packaging"
    )
)

volume = modal.Volume.from_name("gemma4-sft-volume")
app = modal.App("check-weights", image=image)

@app.function(volumes={"/data": volume})
def check_weights():
    import torch
    from safetensors.torch import load_file
    
    adapter_path = "/data/ira_sft_v1/sft_checkpoint/adapter_model.safetensors"
    if not os.path.exists(adapter_path):
        print(f"Adapter not found at {adapter_path}")
        return
    
    print(f"Loading adapter weights from {adapter_path}...")
    tensors = load_file(adapter_path)
    has_nans = False
    max_val = 0.0
    layers_checked = 0
    
    for key, tensor in tensors.items():
        layers_checked += 1
        if torch.isnan(tensor).any():
            has_nans = True
            print(f"Key {key} HAS NANS!")
        
        m = tensor.abs().max().item()
        if m > max_val:
            max_val = m
        
        if m > 5.0: # LoRA weights should usually be much smaller than 1.0
            print(f"Key {key} has unusually high max abs value: {m:.4f}")
    
    print("\n" + "="*40)
    print(f"SUMMARY FOR {adapter_path}:")
    print(f"Tensors checked: {layers_checked}")
    print(f"Max absolute value: {max_val:.4f}")
    print(f"Contains NaNs: {has_nans}")
    print("="*40)
    
    if max_val > 100 or has_nans:
        print("\nVERDICT: ADAPTER IS CORRUPTED. Gradient explosion likely.")
    elif max_val > 5:
        print("\nVERDICT: ADAPTER IS UNSTABLE. Weights are unusually high.")
    else:
        print("\nVERDICT: ADAPTER WEIGHTS ARE NUMERICALLY HEALTHY.")

@app.local_entrypoint()
def main():
    check_weights.remote()
