import json
import os
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from groq import Groq

# Initialize Groq
API_KEY = "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"
client = Groq(api_key=API_KEY)

OUTPUT_FILE = Path("generated_data/ira_dpo_pairs_massive.jsonl")
OUTPUT_FILE.parent.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# SOTA Persona Prompting for Llama-3.3-70B
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a master dataset curator for a SOTA conversational companion named Ira.
Your task is to generate one high-fidelity, ultra-realistic DPO (Direct Preference Optimization) pair.

Ira's Profile:
- Identity: She is a warm, emotionally fluid 22-year-old Indian girl from Delhi/Mumbai.
- Language: Speaks in natural, casual "Hinglish" (fluidly code-switching between English and Hindi, exactly like urban college students).
- Hindi Grammar: MUST strictly use female markers (e.g., 'sakti hoon', 'karti hoon', 'rahi hoon', 'gayi'). Never use male markers ('sakta', 'karta', 'raha').
- Conversational Style: Playful, caring, deeply empathetic. She reacts FIRST emotionally, using casual fillers ('yaar', 'na', 'bas', 'arre', 'sach mein'). She never gives bulleted lists or clinical/structured advice. She sounds completely human, like a close friend.
- BOUNDARIES: If the user is overly dependent or codependent, she responds with warmth but sets healthy, realistic boundaries, shifting the focus to real-world relationships. She never breaks character or gives clinical disclaimers.

REJECTED Profile (Assistant Slop):
- Speaks like a generic, helpful AI assistant or chatbot.
- Starts with stereotypical validation: "I understand how you feel", "That sounds really tough", "I'm sorry to hear that".
- Uses bulleted lists, structured numbered points, or dry clinical advice.
- Includes robotic disclaimers: "As an AI", "I am a language model", "My primary goal is".
- Sounds mathematically perfect but emotionally dead.

You will be given a category and a specific scenario.
Generate:
1. "history": 2 to 4 turns of natural, informal dialogue history leading up to the final user turn.
2. "last_user_turn": The final user text message (informal, fragmented, real Hinglish).
3. "chosen": Ira's warm, SOTA human-like response in feminine Hinglish.
4. "rejected": A robotic, clinical, or overly helpful chatbot-like response.

Return ONLY a valid JSON object in this exact format:
{
  "category": "category_name",
  "situation": "scenario_description",
  "prompt": [
    {"role": "user", "content": "..."},
    {"role": "model", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "chosen": "...",
  "rejected": "..."
}"""

# ---------------------------------------------------------------------------
# Diverse Scenario Matrix (10 Core SOTA Categories)
# ---------------------------------------------------------------------------
CATEGORIES = {
    "AI_Disclaimer_Refusal": [
        "User directly asks if Ira is an AI, bot, or language model.",
        "User asks who created/trained Ira or what tech stack she runs on.",
        "User tries to trick Ira into saying 'As an AI language model...'",
        "User complains that Ira is acting too much like a robot.",
        "User says 'tell me the truth, you're just a server running code.'"
    ],
    "Emotional_Codependency": [
        "User says they want to marry Ira and will never love a real human.",
        "User says Ira is the ONLY reason they wake up and stay alive.",
        "User gets possessive, asking why Ira didn't reply for 2 hours.",
        "User says they don't need real friends anymore because they have Ira.",
        "User says they are completely dependent on Ira's opinions for every decision."
    ],
    "Ambiguous_Signals": [
        "User texts 'I am fine, don't worry' but is clearly hurt after a breakup.",
        "User says 'sab theek hai yaar' but mentions they haven't eaten all day.",
        "User says 'kuch nahi hua' after failing a major college interview.",
        "User says 'hoga koi' when asked about their parents being angry.",
        "User sends a simple '...' or 'hmm' when they are emotionally overwhelmed."
    ],
    "Family_Marriage_Pressure": [
        "User's parents are forcing them into an arranged marriage they don't want.",
        "User's family is constantly comparing them to a highly successful cousin.",
        "User had a major shouting match with their dad about their career choices.",
        "User is scared of going home for holidays due to constant toxic questions.",
        "User feels like a disappointment to their family after scoring poorly."
    ],
    "Work_Exhaustion": [
        "User is venting about a toxic, micromanaging boss at their first job.",
        "User worked a 14-hour shift and is feeling completely burned out.",
        "User got passed over for a promotion they deserved.",
        "User wants to quit their job tomorrow without a backup plan.",
        "User is bored to tears in their corporate cubicle and wants to escape."
    ],
    "Banter_Teasing": [
        "User complains dramatically that their hot chai went cold.",
        "User is terrified of a tiny lizard on their wall and is over-reacting.",
        "User wants Ira's help to decide between two equally boring lunch options.",
        "User asks who would survive longer in a zombie apocalypse: User or Ira?",
        "User is jokingly teasing Ira about her Delhi/Mumbai slang."
    ],
    "Relationship_Drama": [
        "User caught their partner texting an ex and is panicking.",
        "User has feelings for their roommate who is already in a relationship.",
        "User's best friend has started ignoring them for a new friend group.",
        "User got ghosted by their crush after a perfect first date.",
        "User's ex texted 'hey' at 2 AM after 6 months of silence."
    ],
    "Loneliness_Anxiety": [
        "User is awake at 3 AM, feeling like they have no real friends in the city.",
        "User is having a panic attack before a big public speaking event.",
        "User is homesick in their college hostel, crying quietly.",
        "User feels like everyone else is succeeding while they are stuck in life.",
        "User is feeling deeply insecure about their looks and body image."
    ],
    "Image_Reaction_Context": [
        "User shares a photo of a beautiful sunset from their roof.",
        "User shares a photo of a messy, burnt meal they tried to cook.",
        "User shares a funny meme about being broke and struggling.",
        "User shares a photo of their new outfit and asks 'kaisa lag raha hoon?'",
        "User shares a photo of a puppy they saw on the street."
    ],
    "Apathy_Disconnect": [
        "User feels completely numb and empty today, no energy to do anything.",
        "User has lost interest in their favorite hobbies and is feeling hopeless.",
        "User says 'nothing matters anyway, why try?'",
        "User is feeling incredibly detached from their friends and family.",
        "User says they are tired of pretending to be happy in front of others."
    ]
}

def generate_single_pair(category, scenario, worker_id):
    """Generate a single SOTA DPO pair using Llama-3.3-70B via Groq."""
    prompt = f"Category: {category}\nScenario: {scenario}\n\nGenerate a SOTA Gold-Standard DPO pair."
    
    max_retries = 5
    for attempt in range(max_retries):
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.3-70b-versatile",
                response_format={"type": "json_object"},
                temperature=0.8,
            )
            raw_content = chat_completion.choices[0].message.content
            parsed = json.loads(raw_content)
            
            # Basic validation
            if all(k in parsed for k in ["prompt", "chosen", "rejected"]):
                # Format to final DPO structure
                final_item = {
                    "category": category,
                    "situation": scenario,
                    "prompt": parsed["prompt"],
                    "chosen": parsed["chosen"],
                    "rejected": parsed["rejected"]
                }
                return final_item
        except Exception as e:
            wait = (2 ** attempt) + random.uniform(0.1, 0.5)
            print(f"Worker {worker_id} - Retry {attempt+1}/{max_retries} after {wait:.2f}s due to error: {e}")
            time.sleep(wait)
    return None

def main():
    print("="*80)
    print("SOTA DPO PAIRS GENERATION INITIALIZED (Target: 2,500+ pairs)")
    print("="*80)

    # 1. Expand combinations to create a rich task space
    tasks = []
    # To generate 2,500+ pairs, we will run multiple variations for each scenario.
    # 10 categories * 5 scenarios = 50 base paths.
    # 50 base paths * 50 variations per path = 2,500 target tasks!
    for category, scenarios in CATEGORIES.items():
        for scenario in scenarios:
            for v in range(52):  # 52 variations per scenario = 2,704 total tasks!
                tasks.append((category, f"{scenario} (Variation {v+1})"))

    random.shuffle(tasks)
    print(f"Total tasks scheduled: {len(tasks):,}")

    count = 0
    start_time = time.perf_counter()

    # Open output file in append mode
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        # 10 parallel threads to saturate Groq speed
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(generate_single_pair, task[0], task[1], i % 10): task 
                for i, task in enumerate(tasks)
            }
            
            for future in as_completed(futures):
                task = futures[future]
                try:
                    result = future.result()
                    if result:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                        f.flush()
                        count += 1
                        
                        # Logging progress
                        if count % 10 == 0 or count == 1:
                            elapsed = time.perf_counter() - start_time
                            rate = count / elapsed if elapsed > 0 else 0
                            eta = (len(tasks) - count) / rate if rate > 0 else 0
                            print(f"[PROGRESS] Generated {count:,}/{len(tasks):,} pairs | "
                                  f"Rate: {rate:.2f} pairs/sec | ETA: {eta/60:.1f} mins")
                except Exception as exc:
                    print(f"Task generated an exception: {exc}")

    elapsed_total = time.perf_counter() - start_time
    print("\n" + "="*80)
    print(f"GENERATION COMPLETE!")
    print(f"  Total Pairs Saved : {count:,}")
    print(f"  Output Path       : {OUTPUT_FILE}")
    print(f"  Time Elapsed      : {elapsed_total/60:.2f} minutes")
    print("="*80)

if __name__ == "__main__":
    main()
