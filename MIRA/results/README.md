# Checkpoint & Training Results Metadata

Due to the size of the model checkpoints (Gemma-4-E4B 4-bit and full FP16 PEFT adapters), raw weights are stored on the Modal persistent volume `gemma4-sft-volume` under `/data`.

### Checkpoints Generated:
1. **SFT Checkpoint:**
   - **Path on Volume:** `/data/ira_sft_clean/sft_checkpoint`
   - **Hyperparameters:** Epochs=3, LR=2e-4, Batch Size=32 (effective), LoRA Rank=64, Alpha=128.
   - **Primary Objective:** Transformed base model from a standard conversational assistant into a warm, modern Hinglish companion.

2. **DPO / Preference Checkpoint:**
   - **Path on Volume:** `/data/ira_dpo_clean/final_checkpoint`
   - **Hyperparameters:** Epochs=1, LR=2e-6, Beta=0.05, LoRA Rank=64, Alpha=64.
   - **Primary Objective:** Aligned style to eliminate toxic positivity, generic AI refutations, advice-column tone, and mid-word code-switch boundaries.

### Validation Loss Curve Summary:
- **SFT Stage:** Converged smoothly from initial train loss of 1.48 to final validation loss of 0.62.
- **DPO Stage:** Implicit rewards increased consistently for Chosen completions (+0.45 avg margin) while decreasing for Rejected completions (-0.32 avg margin).
