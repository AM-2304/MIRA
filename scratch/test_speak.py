import httpx
import json

url = "https://rumik-ai--ira-tts-service-ttsservice-web-app.modal.run/speak"
payload = {"text": "Hello world!"}

try:
    print("Sending POST request to TTS endpoint...")
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        resp = client.post(url, json=payload)
        print(f"Status Code: {resp.status_code}")
        print(f"Headers: {dict(resp.headers)}")
        print(f"Content Length: {len(resp.content)}")
except Exception as e:
    print(f"Exception: {type(e).__name__} - {e}")
