# Multimodal Ira: Gemma 4 Multimodal Companion

This repository contains the complete end-to-end training, alignment, serving, and evaluation pipeline for **Mira** (Ira's modern, conversational companion counterpart), built on top of Google's multimodal **Gemma 4** base model (`google/gemma-4-E4B-it`).

---

## 📁 Repository Structure
Following the requirements, the repository is structured as follows:
*   `serve/`: Serving infrastructure and FastAPI / Modal endpoints.
    *   `serve_sglang.py`: Main SGLang vision-language LLM CPU/GPU server.
    *   `serve_tts.py`: F5-TTS zero-shot voice synthesis service (positional-parameter robust).
    *   `chat_ui.html`: Minimalist premium web UI featuring low-latency, gesture-unlocked AudioQueue streaming and automatic interrupt.
*   `training/`: Supervised Fine-Tuning (SFT) pipeline.
    *   `sft_pipeline.py`: Training script for SFT adapters on A100.
    *   `modal_sft_job.py`: Scalable SFT runner orchestrated on Modal.
*   `preference/`: Direct Preference Optimization (DPO) pipeline.
    *   `dpo_pipeline.py`: DPO alignment script to eradicate robotic/helper language.
    *   `bulk_generate_dpo.py`: Automated preference pair expansion utility.
*   `evals/`: Specialized companion behavioral evaluation suite.
    *   `evaluator.py`: Testing boundaries on emotional intelligence, safety/refusal rates, and multilingual authenticity.
*   `results/`: Training checkpoints metadata.
*   `final_report.md`: Complete model evaluation, findings, and readiness recommendation.

---

## 🚀 Getting Started & Setup

### 1. Prerequisites
Ensure you have Python 3.11+ and the Modal client installed:
```bash
pip install modal
modal setup
```

### 2. Deployed Inference Serving
Our production setup deploys a multi-node architecture utilizing a shared Modal Volume (`gemma4-sft-volume`) for persistent model weight and reference voice storage.

#### Deploy F5-TTS Voice Service (T4 GPU)
```bash
modal deploy serve/serve_tts.py
```

#### Deploy SGLang Inference Backend & Web UI (A100 GPU + CPU Proxy)
```bash
modal deploy serve/serve_sglang.py
```

After deployment, access the fully interactive web companion at the generated URL:
`https://[your-modal-username]--ira-sglang-service-ui.modal.run`

---

## 🧠 Core Workstreams Implementation

### Workstream 1: Model Strategy
- **Base Model:** `google/gemma-4-E4B-it` selected to leverage native, single-latent-space multimodal processing (understanding user-shared images emotionally rather than procedurally).
- **Training Method:** Quantized Low-Rank Adaptation (QLoRA) with Rank=64, Alpha=128 targeting all attention and MLP layers.
- **Trade-offs:** 4-bit loading achieves sub-second latency and maximizes memory headroom for long multi-turn context support on active chat sessions.

### Workstream 2: SFT (Supervised Fine-Tuning)
- **Data plan:** Focused heavily on code-switching messaging style (Hinglish), transliterated Indic text, and informal slang.
- **Data pipeline:** Implements strict validation and aggressive regex filtering to exclude robotic structures, lists, and assistant jargon before weights are touched.

### Workstream 3: Preference Alignment (DPO)
- **Method:** Direct Preference Optimization (DPO) using `trl.DPOTrainer` with conservative beta setting (`0.05`) and extremely stable learning rate (`2e-6`).
- **Focus:** Eliminating advice-column voice (offering coping mechanisms, bulleted suggestions) and reinforcing a modern, empathetic, conversational companion persona.

### Workstream 4: Serving Stack
- Built on top of high-performance **SGLang** with OpenAI-compatible endpoint compatibility.
- Implements real-time text-streaming, split-second sentence detection, and asynchronous TTS synthesis enqueuing.
- **Client-Side Player:** Dynamic buffer queuing with browser AudioContext unlock on first user gesture to prevent autoplay blocks, supporting instant voice interrupt when the user begins speaking.

### Workstream 5: Evaluation
- Implemented companion-specific test scenarios in `/evals/evaluator.py`.
- Covers safety boundaries, emotional sensitivity ("I'm fine" hidden sadness detection), playful banter, transliteration variations (`bohot` vs `bahut`), and character consistency.

---

## 🔒 Safety & Boundaries
Mira incorporates a rigorous client-side **Two-Strike Lock System**:
1. **Pre-flight NSFW/Slur Check:** All inputs are validated against a curated list of inappropriate content and slurs.
2. **First Strike:** Friendly conversational correction ("Let's not go there").
3. **Second Strike:** Permanent UI lock displaying a sharp boundary message: *"You need help. Touch some grass, you creep!"*