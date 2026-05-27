import subprocess
import json
import time
import os
import shutil
import modal
from pathlib import Path

# Configuration
VOLUME_NAME = "gemma4-sft-volume"
SFT_APP_ID = "ap-yNeJKidtMpGDwRYWWdE1Qn"
DATA_DIR = "/data"

# Setup Modal app for helper operations
image = modal.Image.debian_slim(python_version="3.11")
app = modal.App(name="ira-pipeline-helper", image=image)
volume = modal.Volume.from_name(VOLUME_NAME)

@app.function(volumes={DATA_DIR: volume})
def align_sft_checkpoint():
    """Lightweight function to align checkpoint-300 to sft_checkpoint on volume."""
    import shutil
    import os
    src = "/data/ira_sft_clean/checkpoint-300"
    dst = "/data/ira_sft_clean/sft_checkpoint"
    
    if not os.path.exists(src):
        return f"Error: Source {src} not found on volume!"
        
    if os.path.exists(dst):
        print(f"Removing old destination: {dst}...")
        shutil.rmtree(dst)
        
    print(f"Copying {src} to {dst}...")
    shutil.copytree(src, dst)
    volume.commit()
    return f"Success: Aligned {src} -> {dst}"

def run_local_cmd(cmd, shell=False):
    """Run a local shell command and return stdout/stderr."""
    result = subprocess.run(cmd, capture_output=True, text=True, shell=shell)
    return result.returncode, result.stdout, result.stderr

def check_checkpoint_on_volume():
    """Check if checkpoint-300 exists on the Modal volume using CLI."""
    code, stdout, stderr = run_local_cmd(["python3", "-m", "modal", "volume", "ls", VOLUME_NAME, "/ira_sft_clean"])
    if code != 0:
        print(f"Error checking volume: {stderr}")
        return False
    return "checkpoint-300" in stdout

def main():
    print("="*80)
    print("SELF-DRIVING COMPANION ML PIPELINE ORCHESTRATOR INITIALIZED")
    print("="*80)
    
    # ── Phase 1: Wait for SFT Step 300 ────────────────────────────────────────
    print(f"\n[PHASE 1] Monitoring SFT App ({SFT_APP_ID}) for step 300 checkpoint...")
    start_time = time.perf_counter()
    
    while True:
        if check_checkpoint_on_volume():
            print("\n[SUCCESS] checkpoint-300 detected on Modal volume!")
            break
        
        elapsed_mins = (time.perf_counter() - start_time) / 60
        print(f"  [{elapsed_mins:.1f} mins elapsed] Still training SFT... polling in 30 seconds.")
        time.sleep(30)
        
    # Wait 60s for disk sync
    print("Waiting 60 seconds for absolute volume synchronization and flush...")
    time.sleep(60)
    
    # ── Phase 2: Halt SFT Job ────────────────────────────────────────────────
    print(f"\n[PHASE 2] Gracefully halting SFT app: {SFT_APP_ID}...")
    code, stdout, stderr = run_local_cmd(["python3", "-m", "modal", "app", "stop", SFT_APP_ID])
    if code == 0:
        print("SFT App successfully halted. Compute resources released.")
    else:
        print(f"Warning/Info stopping SFT: {stdout} | {stderr}")
        
    # ── Phase 3: Align Checkpoint Paths ──────────────────────────────────────
    print("\n[PHASE 3] Aligning SFT checkpoint paths remotely...")
    with app.run():
        res = align_sft_checkpoint.remote()
        print(res)
        
    # ── Phase 4: Run DPO Preference Alignment ────────────────────────────────
    print("\n[PHASE 4] Triggering DPO preference training on 2,600 gold pairs...")
    dpo_cmd = "python3 -m modal run /Users/vasu/Documents/GitHub/gemma4-ira-companion/dpo_pipeline.py::run_dpo"
    print(f"Running: {dpo_cmd}")
    
    # Start DPO synchronously in the background (we will stream its output)
    dpo_code = subprocess.call(dpo_cmd, shell=True)
    if dpo_code != 0:
        print("[ERROR] DPO training run failed! Aborting pipeline.")
        return
    print("[SUCCESS] DPO preference training complete!")

    # ── Phase 5: Lossless 16-Bit Weight Merge & Export ────────────────────────
    print("\n[PHASE 5] Compiling clean SFT + DPO weights into 16-bit serving format...")
    merge_cmd = "python3 -m modal run /Users/vasu/Documents/GitHub/gemma4-ira-companion/dpo_pipeline.py::export_model_sglang"
    print(f"Running: {merge_cmd}")
    
    merge_code = subprocess.call(merge_cmd, shell=True)
    if merge_code != 0:
        print("[ERROR] Weight merging failed! Aborting pipeline.")
        return
    print("[SUCCESS] Lossless 16-bit weight export complete!")

    # ── Phase 6: Hot SGLang Server Deploy ────────────────────────────────────
    print("\n[PHASE 6] Hot-deploying production SGLang serving stack on Modal...")
    deploy_cmd = "python3 -m modal deploy /Users/vasu/Documents/GitHub/gemma4-ira-companion/serve_sglang.py"
    print(f"Running: {deploy_cmd}")
    
    deploy_code = subprocess.call(deploy_cmd, shell=True)
    if deploy_code != 0:
        print("[ERROR] SGLang deployment failed!")
        return
        
    print("\n" + "="*80)
    print("PRODUCTION COMPANION PIPELINE COMPLETELY EXECUTED!")
    print("Your SOTA low-latency companion is now served live at your Modal endpoint.")
    print("="*80)

if __name__ == "__main__":
    main()
