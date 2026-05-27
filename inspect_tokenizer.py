import modal
import os

app = modal.App("inspect-tokenizer")
volume = modal.Volume.from_name("gemma4-sft-volume")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("transformers", "accelerate", "huggingface_hub")
)

@app.function(
    image=image,
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
def inspect_tok():
    from transformers import AutoTokenizer
    
    model_name = "google/gemma-4-E4B-it"
    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    out = []
    out.append("=== TOKENIZER INSPECTION ===")
    out.append(f"Model: {model_name}")
    out.append(f"BOS token: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")
    out.append(f"EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    out.append(f"PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    
    out.append("\n=== DEFAULT CHAT TEMPLATE ===")
    out.append(str(tokenizer.chat_template))
    
    # Check vocabulary for turn tokens
    special_candidates = [
        "<start_of_turn>", "<end_of_turn>",
        "<|turn>", "<turn|>",
        "<|im_start|>", "<|im_end|>",
        "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"
    ]
    
    out.append("\n=== SPECIAL TOKEN VOCAB CHECK ===")
    for cand in special_candidates:
        token_id = tokenizer.convert_tokens_to_ids(cand)
        # check if it is in vocab (not equal to unk/ None/ default)
        in_vocab = token_id is not None and token_id != tokenizer.unk_token_id and token_id != 1 and token_id != 3
        out.append(f"Token '{cand}': {'✓ IN VOCAB (ID: ' + str(token_id) + ')' if in_vocab else '✗ NOT IN VOCAB'}")
        
    return "\n".join(out)

if __name__ == "__main__":
    with app.run():
        print(inspect_tok.remote())
