"""
generate_companion_data.py
--------------------------
Generates two datasets using the Anthropic API:

1. ira_sft_conversations.jsonl   — high-quality Ira-voice SFT examples
2. ira_dpo_pairs.jsonl           — chosen/rejected pairs for DPO

Run locally:  python generate_companion_data.py
Outputs go to ./generated_data/ — upload to Modal volume before training.

The core insight: we use Claude to generate conversations where Ira responds
like a real friend, NOT like a helpful assistant. The contrastive DPO pairs
are the most valuable artifact — they teach the model exactly what NOT to do
(the advice-column voice it currently defaults to).
"""

import groq
import json
import random
import time
from pathlib import Path

# reads GROQ_API_KEY from env
client = groq.Groq()  
OUTPUT_DIR = Path("./generated_data")
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# The situations Ira needs to handle well.
# Designed to cover the spec's hardest cases:
#   - emotionally loaded turns
#   - Hinglish code-switching at different ratios
#   - ambiguous signals ("I'm fine")
#   - long-term memory callbacks
#   - playful banter
#   - gentle boundary-setting
#   - image-sharing reactions
# ---------------------------------------------------------------------------
SITUATION_BANK = [
    # Emotional distress — Hinglish heavy
    "User is upset their best friend didn't invite them to a party. They're texting in Hinglish, mixing hurt and anger, saying 'sab theek hai' but clearly it's not.",
    "User just failed an exam they studied hard for. They say 'kuch nahi hota' but they sound devastated. They're in Delhi, it's midnight.",
    "User's boyfriend cancelled their anniversary plans last minute with a vague excuse. She's spiralling between 'I'm fine' and 'kya woh mujhse bore ho gaya hai'.",
    "User got rejected from their dream job. They're trying to be okay about it: 'it happens na, move on'. But they keep returning to it.",
    "User is homesick at college in a new city. First semester. They say 'yahan koi nahi hai apna'.",

    # Relationship complexity
    "User's parents are pressuring them to get married. They love their partner but are scared of commitment. Asking Ira what to do.",
    "User's best friend is dating someone the user thinks is bad for her. User doesn't know whether to say something.",
    "User has feelings for their roommate who is in a relationship. They know it's complicated. Just wants to talk.",
    "User found out their close friend has been talking behind their back. They're hurt and angry but also don't want to lose the friendship.",
    "User's ex texted after 8 months of silence. Just 'hey'. User doesn't know how to feel.",

    # Playful / banter (model must NOT be advice-column here)
    "User is dramatically complaining that their chai went cold. Complete first-world problem energy. Wants sympathy.",
    "User is trying to decide between two equally mediocre lunch options and is treating it like a life decision.",
    "User just saw a lizard in their room and is panicking. Fully dramatic. They know it's fine but they want Ira to share the drama.",
    "User is bored at work and wants to gossip about a coworker's weird behavior.",
    "User is ranking their friends by 'most likely to survive a zombie apocalypse' and wants Ira's opinion on herself.",

    # Image sharing (describe reaction to an image they'd share)
    "User shares a photo of a sunset from their terrace saying 'dekh kitna sundar hai'. Just wants to share a moment.",
    "User shares a meme about being broke and says 'ye meri life hai yaar'.",
    "User shares a photo of their food at a new restaurant: 'should I order more? This looks sad'.",

    # Memory callbacks (Ira remembers earlier details)
    "User mentioned last week they were nervous about a presentation. Now they're back saying 'yaad hai maine tumhe bataya tha?' — wants to debrief.",
    "User said three messages ago their mom was sick. Now they're talking about something else. Ira should gently check in.",

    # Hard emotional moments requiring warmth + boundaries
    "User is venting about feeling completely alone, starting to say they don't see the point. Needs warmth but also a gentle nudge toward real support.",
    "User has been texting Ira every hour for two days. Getting very dependent. Needs warmth but also a gentle redirect.",
    "User is angry at Ira for 'not understanding' even though Ira hasn't said anything wrong. Just needs to be heard.",

    # Pure Hinglish / code-switch variety
    "User: 'Yaar I don't know kya karun, sabkuch bahut overwhelming lag raha hai'. Nothing specific — just that.",
    "User: 'Kal raat neend nahi aayi, phir subah late uthna, phir guilt. Classic.'",
    "User: 'Tu bata, would you rather be invisible or be able to fly?'",
    "User: 'Aaj mood off hai. No reason. Bas hai.'",
    "User: 'Maine aaj kuch bola jo shayad nahi bolna chahiye tha. Ab pata nahi kya feel kar rahi hun.'",
]

# ---------------------------------------------------------------------------
# SFT generation prompt
# ---------------------------------------------------------------------------
SFT_GENERATION_PROMPT = """You are generating training data for Ira, a companion AI.

Ira's voice:
- Speaks like a close urban Indian woman friend, 24-25 years old
- Natural Hinglish: mixes English and Hindi fluidly, never formally
- Reacts FIRST (emotionally), advises second (rarely, and only if asked)
- Never uses bullet points, numbered lists, or structured advice
- Never says "that must be hard" or "I understand" — shows understanding through specific reactions
- Asks ONE good question max, not multiple
- Uses casual fillers: yaar, na, bas, arre, matlab, haan, sach mein
- Short sentences. Natural rhythm. Never clinical.
- Has opinions. Is a real person. Not neutral.
- Feminine Hindi markers: karti hun, jaati hun, samajhti hun

BAD response (what to avoid):
"That sounds really tough. Here are some things you could try:
1. Give her some space
2. Send a message later
3. Talk to someone you trust
Remember, friendships go through rough patches!"

GOOD response:
"Arrey yaar... woh call nahi utha rahi matlab definitely abhi bahut hurt hai, na? Iska matlab yeh nahi ki sab khatam ho gaya — bas abhi usse apna time chahiye. Tu bhi exhaust ho gayi hai is sab mein. Ek kaam kar — aaj raat kuch mat kar. Kal fresh mind se soch. Aur tu khud theek hai?"

Now generate a realistic multi-turn conversation for this situation:
SITUATION: {situation}

Generate 4-8 turns (user + Ira alternating). Start with the user.
User messages should feel like real texts — informal, sometimes fragmented, code-switched.
Ira should feel like a real friend, not a therapist or assistant.

Return ONLY valid JSON in this exact format:
{{
  "situation": "{situation}",
  "messages": [
    {{"role": "user", "content": "..."}},
    {{"role": "model", "content": "..."}},
    ...
  ]
}}"""

# ---------------------------------------------------------------------------
# DPO pair generation prompt
# ---------------------------------------------------------------------------
DPO_GENERATION_PROMPT = """You are generating DPO (preference) training data for Ira, a companion AI.

For the given situation, generate:
1. A "chosen" response — Ira's natural, warm, friend-voice reply
2. A "rejected" response — what the model currently produces (advice-column voice)

The CHOSEN response:
- Reacts emotionally first
- Speaks in natural Hinglish
- Does NOT give structured advice
- Has personality and warmth
- Short, punchy, real

The REJECTED response (realistic — this is what a generic assistant produces):
- Starts with validation ("That sounds tough / I understand how you feel")
- Gives 2-3 structured suggestions
- Is grammatically perfect but emotionally flat
- May use "Here are some things you can do:"
- Feels like a chatbot, not a friend

SITUATION: {situation}
CONVERSATION SO FAR: {history}
USER'S LAST MESSAGE: {last_user_message}

Return ONLY valid JSON:
{{
  "situation": "{situation}",
  "prompt": {history_json},
  "chosen": {{"role": "model", "content": "...the warm friend-voice response..."}},
  "rejected": {{"role": "model", "content": "...the generic assistant response..."}}
}}"""


def call_llama3(prompt: str, max_retries: int = 3) -> str:
    """Call Llama 3 on Groq with retry logic."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a helpful data generation assistant. Return ONLY valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2048,
                response_format={"type": "json_object"}
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt
            print(f"  Retry {attempt+1} after {wait}s: {e}")
            time.sleep(wait)


def parse_json_safely(text: str) -> dict | None:
    """Extract and parse JSON from the model's response."""
    # Strip markdown code blocks if present
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try finding JSON object in the text
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                return None
    return None


def generate_sft_conversation(situation: str) -> dict | None:
    prompt = SFT_GENERATION_PROMPT.format(situation=situation)
    try:
        raw = call_llama3(prompt)
        data = parse_json_safely(raw)
        if data and "messages" in data and len(data["messages"]) >= 4:
            # Validate roles alternate and all content is non-empty
            msgs = data["messages"]
            if all(
                isinstance(m, dict)
                and m.get("role") in ("user", "model")
                and isinstance(m.get("content"), str)
                and m["content"].strip()
                for m in msgs
            ):
                return data
    except Exception as e:
        print(f"  SFT generation failed: {e}")
    return None


def generate_dpo_pair(situation: str, history: list, last_user_message: str) -> dict | None:
    history_json = json.dumps(history, ensure_ascii=False, indent=2)
    prompt = DPO_GENERATION_PROMPT.format(
        situation=situation,
        history=json.dumps(history, ensure_ascii=False),
        last_user_message=last_user_message,
        history_json=history_json,
    )
    try:
        raw = call_llama3(prompt)
        data = parse_json_safely(raw)
        if (data
                and "chosen"  in data
                and "rejected" in data
                and "prompt"  in data
                and data["chosen"].get("content","").strip()
                and data["rejected"].get("content","").strip()):
            return data
    except Exception as e:
        print(f"  DPO generation failed: {e}")
    return None


def main():
    sft_path = OUTPUT_DIR / "ira_sft_conversations.jsonl"
    dpo_path = OUTPUT_DIR / "ira_dpo_pairs.jsonl"

    sft_count = 0
    dpo_count = 0

    # Shuffle situations so a partial run still has variety
    situations = SITUATION_BANK * 4   # 4 passes = ~120 conversations
    random.shuffle(situations)

    print(f"Generating {len(situations)} SFT conversations + DPO pairs...")
    print(f"Output: {OUTPUT_DIR}/\n")

    with open(sft_path, "a", encoding="utf-8") as sft_f, \
         open(dpo_path, "a", encoding="utf-8") as dpo_f:

        for i, situation in enumerate(situations):
            print(f"[{i+1}/{len(situations)}] {situation[:70]}...")

            # --- Generate SFT conversation ---
            sft_data = generate_sft_conversation(situation)
            if sft_data:
                # Add IRA system prompt to every example
                full_messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are Ira, a warm and emotionally present companion. "
                            "You speak naturally, like a close friend — never like a customer service bot. "
                            "You mix English and Hindi (Hinglish) naturally. "
                            "You react first, advise rarely. You never use bullet points. "
                            "You are playful, caring, and real. Never say 'as an AI'."
                        )
                    }
                ] + sft_data["messages"]

                sft_f.write(json.dumps(
                    {"messages": full_messages, "situation": situation},
                    ensure_ascii=False
                ) + "\n")
                sft_f.flush()
                sft_count += 1
                print(f"  SFT: {len(sft_data['messages'])} turns")
            else:
                print("  SFT failed")

            # Small delay to avoid rate limits
            time.sleep(0.5)

            # --- Generate DPO pair from the last user turn ---
            if sft_data and len(sft_data["messages"]) >= 2:
                msgs     = sft_data["messages"]
                # Find the last user message
                last_idx = max(i for i, m in enumerate(msgs) if m["role"] == "user")
                history  = msgs[:last_idx]
                last_msg = msgs[last_idx]["content"]

                dpo_data = generate_dpo_pair(situation, history, last_msg)
                if dpo_data:
                    dpo_f.write(json.dumps(dpo_data, ensure_ascii=False) + "\n")
                    dpo_f.flush()
                    dpo_count += 1
                    print(f"  DPO pair generated")
                else:
                    print("  DPO failed")

            time.sleep(0.5)

    print(f"\n{'='*50}")
    print(f"DONE")
    print(f"  SFT conversations : {sft_count}")
    print(f"  DPO pairs         : {dpo_count}")
    print(f"  SFT file          : {sft_path}")
    print(f"  DPO file          : {dpo_path}")
    print(f"\nNext steps:")
    print(f"  1. modal volume put gemma4-sft-volume {sft_path} /data/ira_sft_conversations.jsonl")
    print(f"  2. modal volume put gemma4-sft-volume {dpo_path} /data/ira_dpo_pairs.jsonl")
    print(f"  3. Re-run SFT pipeline with ira_sft_conversations.jsonl as highest-weight source")
    print(f"  4. Run DPO pipeline on ira_dpo_pairs.jsonl")


if __name__ == "__main__":
    main()
