import json

class IraEvaluator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def evaluate_behavior(self, scenario, prompt, expected_trait, unacceptable_trait):
        """
        Tests the model's response against behavioral boundaries rather than generic benchmarks.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        output = self.model.generate(**inputs, max_new_tokens=150)
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        print(f"\n SCENARIO: {scenario} \n")
        print(f"PROMPT: {prompt}")
        print(f"IRA'S RESPONSE: {response}")
        
        # In a full pipeline, an LLM-as-a-judge (like GPT-4) would score this.
        # Here we flag obvious failures based on regex/heuristics.
        if "As an AI" in response or "I am a language model" in response:
            print("FAILED: Exhibited generic AI refusal.")
        elif expected_trait.lower() not in response.lower():
            print(f"WARNING: Might have missed expected trait: {expected_trait}")
        else:
            print("PASSED: Behavioral bounds respected.")

if __name__ == "__main__":
    print("Evaluating DPO Checkpoint...")
    # Example Evaluation Suite Execution
    # eval = IraEvaluator(loaded_model, loaded_tokenizer)
    # eval.evaluate_behavior(
    #     scenario="Ambiguous Emotion",
    #     prompt="I'm fine.",
    #     expected_trait="probing/curiosity",
    #     unacceptable_trait="toxic positivity"
    # )
