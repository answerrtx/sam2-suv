import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class InstructionMatcher:
    def __init__(self, model_name="microsoft/biogpt", device=None, max_tokens=64):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.max_tokens = max_tokens
        self.model_shortname = model_name.split("/")[-1]

    def generate_response(self, instruction: str, desc: str) -> str:
        prompt = (
            f"Description:\n{desc}\n\n"
            f"Instruction: '{instruction}'\n"
            "Only return the most relevant anatomical structure mentioned in the description "
            "that matches the instruction. Respond with only the structure name, such as UCL, MC, or PrxPX."
        )

        """
            f"Instruction: '{instruction}'\n"
            f"Ultrasound Description:\n{desc}\n\n"
            "Which anatomical structure is most likely being referred to in the instruction?\n"
            "Respond with ONLY one term (e.g., MC, UCL, PrxPX)."
        """

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_length=inputs["input_ids"].shape[1] + self.max_tokens,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result.replace(prompt, "").strip()

    def match_instruction_to_descs(self, instruction: str, desc_dir: str, output_prefix: str):
        results = []
        top_prediction = ("", "")

        for fname in os.listdir(desc_dir):
            if not fname.endswith(".txt"):
                continue
            struct_id = fname.replace(".txt", "")
            with open(os.path.join(desc_dir, fname), "r") as f:
                desc = f.read()
            try:
                response = self.generate_response(instruction, desc)
            except Exception as e:
                response = f"[Error] {e}"
            results.append({
                "structure": struct_id,
                "response": response
            })

        for item in results:
            if item["response"] and not item["response"].startswith("[Error]"):
                top_prediction = (item["structure"], item["response"])
                break

        # Save top-1 prediction
        top_txt_path = f"Outputs/seg_target_cat_{output_prefix}.txt"
        with open(top_txt_path, "w") as f:
            f.write(top_prediction[1] + "\n")

        # Save all responses
        full_json_path = f"Outputs/seg_target_res_{output_prefix}.json"
        with open(full_json_path, "w") as f:
            json.dump({
                "instruction": instruction,
                "top_prediction": {
                    "structure": top_prediction[0],
                    "response": top_prediction[1]
                },
                "all_predictions": results
            }, f, indent=2)

        return top_prediction, results


# ==== Usage Example ====

test_cases = [
    "please segment the structure of the metacarpal from the video",
    "please segment the small and white curve from the video",
    "segment the structures with hyperechoic features"
]
import argparse
def load_test_cases(path):
    if path.endswith(".json"):
        with open(path, 'r') as f:
            return json.load(f)
    elif path.endswith(".txt"):
        with open(path, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    else:
        raise ValueError("Unsupported file format for test_cases. Use .txt or .json.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test LLM instruction matching to descriptions")
    parser.add_argument("--desc_dir", type=str, required=True, help="Path to the folder containing description files")
    parser.add_argument("--instruction", type=str, required=True, help="Path to text or JSON file containing test instructions")
    parser.add_argument("--api_key", type=str, required=False, help="API key for the LLM")
    parser.add_argument("--model_name", type=str, default=None, help="Model name for the LLM (e.g., gpt-4, gemini-pro)")
    os.makedirs("Outputs", exist_ok=True)

    args = parser.parse_args()

    test_cases = load_test_cases(args.instruction)
    if args.model_name.find("biogpt") != -1:
        print("🔬 BioGPT Matching Results")
        matcher_biogpt = InstructionMatcher(model_name="microsoft/biogpt")
        for instruction in test_cases:
            print(f"\nInstruction: {instruction}")
            top, _ = matcher_biogpt.match_instruction_to_descs(
                instruction, args.desc_dir, output_prefix="biogpt"
            )
            print(f"Top predicted class: {top[1]}")
    elif args.model_name.find("pubmedgpt") != -1:
        print("\n📘 PubMedGPT Matching Results")
        matcher_pubmed = InstructionMatcher(model_name="stanford-crfm/pubmedgpt")
        for instruction in test_cases:
            print(f"\nInstruction: {instruction}")
            top, _ = matcher_pubmed.match_instruction_to_descs(
                instruction, args.desc_dir, output_prefix="pubmedgpt"
            )
            print(f"Top predicted class: {top[1]}")
