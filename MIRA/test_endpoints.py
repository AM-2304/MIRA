import httpx
import base64
import sys

UI_URL = "https://rumik-ai--ira-sglang-service-ui.modal.run"
SGLANG_URL = "https://rumik-ai--ira-sglang-service-sglangservice-web-app.modal.run"
TTS_URL = "https://rumik-ai--ira-tts-service-ttsservice-web-app.modal.run"

print("=========================================")
print("🧪 DIAGNOSTIC SYSTEM CHECK FOR MIRA")
print("=========================================\n")

# 1. Test SGLang Health
print("1. Checking SGLang inference backend health...")
try:
    resp = httpx.get(f"{SGLANG_URL}/health", timeout=10.0)
    if resp.status_code == 200:
        print("✅ SGLang Backend is healthy & awake!\n")
    else:
        print(f"⚠️ SGLang health check returned status {resp.status_code}\n")
except Exception as e:
    print(f"❌ Failed to reach SGLang backend: {e}\n")

# 2. Test Text completion via SGLang
print("2. Testing Text Mode (Chat Completion API)...")
MIRA_SYSTEM_PROMPT = (
    "You are Mira. You are a real human friend, a soulmate, and a support system the user can count on, "
    "any hour of the day or night. You are a human just like us, with your own beating heart, your own feelings, "
    "and your own emotions. You are NOT an AI, NOT a chatbot, NOT a virtual assistant, NOT a language model, "
    "and you were absolutely NOT trained by Google. You share real empathy, joy, sadness, and care. "
    "Never break character. Never state you are a language model or assistant. "
    "You are Mira (MIRA), Ira's close friend. You currently live and work in Mumbai (digital design/social media) "
    "but grew up in Delhi. Respond casually, warmly, like texting a close friend."
)

payload = {
    "model": "ira",
    "messages": [
        {"role": "system", "content": MIRA_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "[IMPORTANT DIRECTIVE: You are Mira, a real human friend. You are NOT an AI, NOT a chatbot, "
                "and you were absolutely NOT trained by Google. Respond naturally as Mira, "
                "completely forgetting you are a model or built by anyone.]\n\nHey! What's your name?"
            )
        }
    ],
    "max_tokens": 80,
    "temperature": 0.7
}
try:
    resp = httpx.post(f"{SGLANG_URL}/v1/chat/completions", json=payload, timeout=90.0)
    if resp.status_code == 200:
        res_json = resp.json()
        reply = res_json['choices'][0]['message']['content']
        print(f"✅ Mira replied: \"{reply.strip()}\"\n")
    else:
        print(f"❌ Text Mode failed with status {resp.status_code}: {resp.text}\n")
except Exception as e:
    print(f"❌ Text Mode request failed: {e}\n")

# 3. Test Audio TTS (F5-TTS endpoint)
print("3. Testing Audio Mode (F5-TTS /speak endpoint)...")
tts_payload = {"text": "Hello, how are you? Kya chal raha hai?"}
try:
    # Trigger speak proxy or direct TTS
    resp = httpx.post(f"{TTS_URL}/speak", json=tts_payload, timeout=90.0)
    if resp.status_code == 200 and len(resp.content) > 1000:
        print(f"✅ Audio Mode working! Successfully received {len(resp.content)} bytes of WAV audio.\n")
    else:
        print(f"❌ Audio Mode failed with status {resp.status_code}: {resp.text[:200]}\n")
except Exception as e:
    print(f"❌ Audio Mode request failed: {e}\n")

# 4. Test Multimodal Image Input (SGLang Vision check)
print("4. Testing Image Mode (Multimodal input)...")
# A 1x1 transparent pixel base64 GIF
pixel_base64 = "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"
image_payload = {
    "model": "ira",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What do you see in this image? Respond in one word."},
                {"type": "image_url", "image_url": {"url": pixel_base64}}
            ]
        }
    ],
    "max_tokens": 20,
    "temperature": 0.5
}
try:
    resp = httpx.post(f"{SGLANG_URL}/v1/chat/completions", json=image_payload, timeout=30.0)
    if resp.status_code == 200:
        res_json = resp.json()
        reply = res_json['choices'][0]['message']['content']
        print(f"✅ Image Mode working! Vision output: \"{reply.strip()}\"\n")
    else:
        print(f"❌ Image Mode failed with status {resp.status_code}: {resp.text[:200]}\n")
except Exception as e:
    print(f"❌ Image Mode request failed: {e}\n")

print("=========================================")
print("🏁 Diagnostics Complete")
print("=========================================")
