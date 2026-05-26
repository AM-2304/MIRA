import modal
import os

app = modal.App("check-paths")
volume = modal.Volume.from_name("gemma4-sft-volume")

@app.function(volumes={"/data": volume})
def check_paths():
    exists_dict = {}
    paths = [
        "/data/ira_sft_v1",
        "/data/ira_sft_v1/sft_checkpoint",
        "/data/ira_dpo_v2",
        "/data/ira_dpo_v2/dpo_checkpoint",
        "/data/dpo_results",
        "/data/ira_final_sglang_16bit",
        "/data/ira_final_sglang_16bit/config.json",
        "/data/results",
        "/data/sft_results",
    ]
    for p in paths:
        exists_dict[p] = os.path.exists(p)
        
    out = []
    out.append("=== PATH CHECK ===")
    for p, exists in exists_dict.items():
        out.append(f"{p}: {'✓ EXISTS' if exists else '✗ MISSING'}")
    
    # Also list top level directories
    try:
        top_dirs = os.listdir("/data")
        out.append("\nTop-level entries in /data:")
        for td in top_dirs:
            p = os.path.join("/data", td)
            is_dir = os.path.isdir(p)
            out.append(f"  {td} {'(Dir)' if is_dir else '(File)'}")
    except Exception as e:
        out.append(f"Error listing /data: {e}")
        
    return "\n".join(out)

if __name__ == "__main__":
    with app.run():
        print(check_paths.remote())
