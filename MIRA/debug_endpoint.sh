#!/bin/bash
# debug_endpoint.sh
# Run this in your terminal to diagnose the SGLang endpoint step by step.
# Usage: bash debug_endpoint.sh

BASE="https://rumik-ai--ira-sglang-service-serve.modal.run"

echo ""
echo "=== STEP 1: Is the SGLang server reachable at all? ==="
curl -s -o /dev/null -w "HTTP status: %{http_code}\n" "$BASE/health"

echo ""
echo "=== STEP 2: Does SGLang list available models? ==="
curl -s "$BASE/v1/models" | python3 -m json.tool 2>/dev/null || \
  curl -s "$BASE/v1/models"

echo ""
echo "=== STEP 3: Minimal chat completion (raw response) ==="
curl -s -X POST "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ira",
    "messages": [{"role": "user", "content": "hey"}],
    "max_tokens": 50
  }' | python3 -m json.tool 2>/dev/null || \
  curl -s -X POST "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"ira","messages":[{"role":"user","content":"hey"}],"max_tokens":50}'

echo ""
echo "=== STEP 4: Streaming chat completion ==="
curl -s -X POST "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ira",
    "messages": [{"role": "user", "content": "hey"}],
    "max_tokens": 50,
    "stream": true
  }'

echo ""
echo "Done. Paste the output in your agent so it can diagnose."