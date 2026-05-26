import json
import os

input_file = "generated_data/ira_dpo_pairs.jsonl"
output_file = "generated_data/ira_dpo_pairs_cleaned.jsonl"

# Mapping male to female markers
mapping = {
    "raha hoon": "rahi hoon",
    "sakta hoon": "sakti hoon",
    "karta hoon": "karti hoon",
    "chala gaya": "chali gayi",
    "aa gaya": "aa gayi",
    "samajh sakta": "samajh sakti",
    "bolta hoon": "bolti hoon",
    "sochta hoon": "sochti hoon",
    "kar raha": "kar rahi",
    "kar sakta": "kar sakti",
    "hoon na?": "hoon na?", # Neutral
}

cleaned_data = []

if os.path.exists(input_file):
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            # Focus on cleaning the 'chosen' response
            chosen_text = item["chosen"]["content"]
            
            for male, female in mapping.items():
                chosen_text = chosen_text.replace(male, female)
            
            item["chosen"]["content"] = chosen_text
            cleaned_data.append(item)

    with open(output_file, "w", encoding="utf-8") as f:
        for item in cleaned_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ Cleaned {len(cleaned_data)} pairs. Saved to {output_file}")
else:
    print("❌ Input file not found!")
