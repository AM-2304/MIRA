import json
import os
import time
from groq import Groq

# Initialize Groq
client = Groq(api_key="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")

INPUT_FILE = "generated_data/ira_dpo_pairs_cleaned.jsonl"
OUTPUT_FILE = "generated_data/ira_dpo_pairs_v2.jsonl"

SYSTEM_PROMPT = """You are a master of Hinglish and feminine Hindi persona. 
Your task is to review and POLISH DPO pairs for 'Ira', a warm, emotionally intelligent female companion.

RULES for 'Chosen' responses:
1. MUST use feminine markers: 'rahi hoon', 'karti hoon', 'sakti hoon', 'gayi'. Never use 'raha/karta/sakta'.
2. MUST sound like a real 22-year-old girl from Delhi/Mumbai - mix English and Hindi naturally.
3. NO robotic advice. Be a friend. React with empathy first.
4. Keep the 'Rejected' responses as they are (robotic/formal).

Format your output as a single JSON object with 'chosen' and 'rejected' keys."""

def upgrade_pair(pair):
    prompt = f"Situation: {pair.get('situation', 'General chat')}\n"
    prompt += f"Context: {json.dumps(pair['prompt'][-2:], ensure_ascii=False)}\n"
    prompt += f"Current Chosen (Needs Polish): {pair['chosen']['content']}\n"
    prompt += f"Current Rejected: {pair['rejected']['content']}\n"
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Upgrade this DPO pair to 'Gold Standard' quality. Return ONLY valid JSON.\n\n{prompt}"}
            ],
            model="llama-3.3-70b-versatile",
            response_format={"type": "json_object"}
        )
        new_data = json.loads(chat_completion.choices[0].message.content)
        pair["chosen"]["content"] = new_data["chosen"]
        # Keep rejected as is or use the one from LLM if it's better robotic
        if "rejected" in new_data:
            pair["rejected"]["content"] = new_data["rejected"]
        return pair
    except Exception as e:
        print(f"Error upgrading: {e}")
        return pair

# 1. POLISH EXISTING
print(f"Polishing existing {INPUT_FILE}...")
upgraded_data = []
with open(INPUT_FILE, "r") as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        item = json.loads(line)
        print(f"  [{i+1}/{len(lines)}] Polishing...")
        upgraded_data.append(upgrade_pair(item))
        time.sleep(0.5) # Rate limit safety

# 2. SAVE
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for item in upgraded_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Upgrade complete! Total pairs: {len(upgraded_data)}")
