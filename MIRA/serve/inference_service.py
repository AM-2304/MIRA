from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional, List
import torch
from unsloth import FastVisionModel

app = FastAPI(title="Ira Companion Inference Service")

# Load final DPO aligned model
MODEL_PATH = "/data/dpo_results/final_dpo"
model = None
tokenizer = None

@app.on_event("startup")
async def load_model():
    global model, tokenizer
    # Using Unsloth FastVisionModel for multimodal deployability
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=MODEL_NAME, max_seq_length=4096, load_in_4bit=True
    )
    FastVisionModel.for_inference(model)

class ChatTurn(BaseModel):
    role: str
    content: str

class InferenceRequest(BaseModel):
    history: List[ChatTurn]
    memory_context: Optional[str] = None
    temperature: float = 0.6
    top_p: float = 0.9

@app.post("/chat")
async def chat(request: InferenceRequest):
    """
    Handles text-only multi-turn chat with memory injection.
    """
    system_prompt = "You are Ira. Warm, perceptive, uses natural Hinglish."
    if request.memory_context:
        system_prompt += f"\nMEMORY: {request.memory_context}"
        
    # Format using tokenizer chat template
    prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt}] + request.history,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=200, temperature=request.temperature, top_p=request.top_p)
    response_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    return {"role": "model", "content": response_text}

@app.post("/chat_multimodal")
async def chat_multimodal(image: UploadFile = File(...), prompt: str = Form(...), memory: str = Form(None)):
    """
    Handles image + text input.
    """
    # 1. Process image bytes (PIL)
    # 2. Format multimodal prompt
    # 3. Generate response with vision model
    return {"role": "model", "content": "Multimodal response processing..."}
