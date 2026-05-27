import json
import os

INPUT_FILE = "generated_data/ira_dpo_pairs_expanded.jsonl"
OUTPUT_FILE = "generated_data/ira_dpo_pairs_fixed.jsonl"

fixed_count = 0
total_count = 0

with open(INPUT_FILE, "r", encoding="utf-8") as f, open(OUTPUT_FILE, "w", encoding="utf-8") as out:
    for line in f:
        line = line.strip()
        if not line: continue
        try:
            item = json.loads(line)
            prompt = item.get("prompt", [])
            
            # Ensure prompt ends with a 'user' role
            while prompt and prompt[-1].get("role") == "model":
                prompt.pop()
            
            if not prompt or prompt[-1].get("role") != "user":
                # If we emptied it or it's still wrong, skip
                continue
                
            item["prompt"] = prompt
            out.write(json.dumps(item, ensure_ascii=False) + "\n")
            fixed_count += 1
        except:
            continue
        total_count += 1

print(f"✅ Repaired {fixed_count} out of {total_count} pairs. Saved to {OUTPUT_FILE}")
