import json
import os

INPUT_FILE = "generated_data/ira_dpo_pairs_expanded.jsonl"
OUTPUT_FILE = "generated_data/ira_dpo_pairs_final.jsonl"

final_count = 0

with open(INPUT_FILE, "r", encoding="utf-8") as f, open(OUTPUT_FILE, "w", encoding="utf-8") as out:
    for line in f:
        line = line.strip()
        if not line: continue
        try:
            item = json.loads(line)
            raw_prompt = item.get("prompt", [])
            
            new_prompt = []
            for i, msg in enumerate(raw_prompt):
                # Detect role: alternate user/model
                role = "user" if i % 2 == 0 else "model"
                
                content = ""
                if isinstance(msg, dict):
                    content = msg.get("content") or msg.get("message") or str(msg)
                else:
                    content = str(msg)
                
                new_prompt.append({"role": role, "content": content})
            
            # Ensure it ends with user
            while new_prompt and new_prompt[-1]["role"] == "model":
                new_prompt.pop()
            
            if not new_prompt:
                continue

            item["prompt"] = new_prompt
            
            # Handle chosen/rejected if they are dicts
            def get_content(val):
                if isinstance(val, dict):
                    return val.get("content") or str(val)
                return str(val)
            
            item["chosen"] = get_content(item["chosen"])
            item["rejected"] = get_content(item["rejected"])
            
            out.write(json.dumps(item, ensure_ascii=False) + "\n")
            final_count += 1
        except Exception as e:
            print(f"Error: {e}")
            continue

print(f"✅ Final Dataset Ready! {final_count} robust pairs saved to {OUTPUT_FILE}")
