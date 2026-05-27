import modal
import os
import json

app = modal.App("inspect-json-configs")
volume = modal.Volume.from_name("gemma4-sft-volume")

@app.function(volumes={"/data": volume})
def inspect_configs():
    paths = {
        "SFT Config": "/data/ira_sft_v1/sft_checkpoint/adapter_config.json",
        "DPO Config": "/data/ira_dpo_v2/dpo_checkpoint/adapter_config.json",
    }
    
    out = []
    for name, p in paths.items():
        out.append(f"=== {name} ({p}) ===")
        if os.path.exists(p):
            try:
                with open(p, "r") as f:
                    config_data = json.load(f)
                    # format json
                    out.append(json.dumps(config_data, indent=2))
            except Exception as e:
                out.append(f"Error reading: {e}")
        else:
            out.append("✗ FILE MISSING")
        out.append("-" * 40 + "\n")
        
    return "\n".join(out)

if __name__ == "__main__":
    with app.run():
        print(inspect_configs.remote())
