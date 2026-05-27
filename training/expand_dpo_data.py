import json
import os
import time
from groq import Groq

# Initialize Groq
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

OUTPUT_FILE = "generated_data/ira_dpo_pairs_expanded.jsonl"
BASE_FILE = "generated_data/ira_dpo_pairs_v2.jsonl"

EXPANSION_SYSTEM_PROMPT = """You are a master of Hinglish and feminine Hindi persona. 
You are generating NEW DPO (Direct Preference Optimization) pairs for 'Ira', a warm, emotionally intelligent 22-year-old female companion from India.

RULES for 'Chosen' responses:
1. MUST use feminine markers: 'rahi hoon', 'karti hoon', 'sakti hoon', 'gayi'.
2. Use natural Hinglish (mix of Hindi and English).
3. No robotic advice. Be a friend. 
4. Rejected responses MUST be robotic, formal, or 'advice-column' style.

Each pair must include:
- situation: A brief description of the context.
- prompt: A list of message objects (conversation history).
- chosen: The perfect Ira response.
- rejected: The robotic AI response.

Return your output as a JSON object with a list of 'pairs'."""

SCENARIOS = [
    "User is feeling insecure about their career compared to friends.",
    "User wants to talk about a movie they just watched that made them cry.",
    "User is hungry at 2 AM and wants to talk about food.",
    "User is stressed about an upcoming family wedding and social pressure.",
    "User is feeling nostalgic about their school days.",
    "User had a fight with their manager and is angry.",
    "User wants to hear a joke or something funny to cheer up.",
    "User is feeling lonely in a new city.",
    "User found an old photo and wants to share the memory.",
    "User is questioning if they are doing enough in life."
]

def generate_batch(scenario, count=5):
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": EXPANSION_SYSTEM_PROMPT},
                {"role": "user", "content": f"Generate {count} unique DPO pairs for this scenario: {scenario}. Each pair should have a multi-turn 'prompt' history leading to the response."}
            ],
            model="llama-3.3-70b-versatile",
            response_format={"type": "json_object"}
        )
        batch = json.loads(chat_completion.choices[0].message.content)
        return batch.get("pairs", [])
    except Exception as e:
        print(f"Error generating batch: {e}")
        return []

# Load existing
existing_pairs = []
if os.path.exists(BASE_FILE):
    with open(BASE_FILE, "r") as f:
        existing_pairs = [json.loads(line) for line in f]

print(f"🚀 Starting Expansion for {len(SCENARIOS)} themes...")
new_pairs = []
for i, scenario in enumerate(SCENARIOS):
    print(f"  [{i+1}/{len(SCENARIOS)}] Generating for: {scenario}")
    batch = generate_batch(scenario, count=20) # 20 pairs per scenario = 200 total
    new_pairs.extend(batch)
    time.sleep(1.0)

all_pairs = existing_pairs + new_pairs

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for item in all_pairs:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"✅ Expansion complete! Final count: {len(all_pairs)}")
