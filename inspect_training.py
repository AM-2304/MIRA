import modal
import os
import json

app = modal.App("inspect-training")
volume = modal.Volume.from_name("gemma4-sft-volume")

@app.function(volumes={"/data": volume})
def inspect():
    paths = {
        "SFT Training Config": "/data/ira_sft_v1/training_config.json",
        "SFT Trainer State": "/data/ira_sft_v1/sft_checkpoint/trainer_state.json",
        "DPO Config": "/data/ira_dpo_v2/dpo_config.json",
        "DPO Trainer State": "/data/ira_dpo_v2/trainer_state.json",
    }
    
    out = []
    for name, p in paths.items():
        out.append(f"=== {name} ({p}) ===")
        if os.path.exists(p):
            try:
                with open(p, "r") as f:
                    data = json.load(f)
                    # For trainer_state, only print the last 20 steps to save space
                    if "log_history" in data:
                        history = data["log_history"]
                        out.append(f"Total log steps: {len(history)}")
                        out.append("Last 15 log steps:")
                        for step in history[-15:]:
                            out.append(json.dumps(step))
                        # remove history to print the rest of the metadata
                        del data["log_history"]
                        out.append("Metadata:")
                        out.append(json.dumps(data, indent=2))
                    else:
                        out.append(json.dumps(data, indent=2))
            except Exception as e:
                out.append(f"Error reading: {e}")
        else:
            out.append("✗ FILE MISSING")
        out.append("-" * 40 + "\n")
        
    return "\n".join(out)

if __name__ == "__main__":
    with app.run():
        print(inspect.remote())
