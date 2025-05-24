


import os
import re
import json
import requests
import openai
from typing import Optional

gpt_api_key ="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"
gemini_api_key="AIzaSyD6aRvxrVlXHfqQfjCYDSOOfVeogf9RsVs"
query_folder = ""

class LLMRouter:
    def __init__(self, provider: str, api_key: str, model: Optional[str] = None):
        self.provider = provider.lower()
        self.api_key = api_key
        self.model = model
        with open("color_map_full.json", "r") as f:
            entries = json.load(f)
        self.fullname_to_abbr = {
            entry["Full Name"].lower(): entry["Abbreviation"]
            for entry in entries
        }
        self.all_fullnames = list(self.fullname_to_abbr.keys())

    def query(self, prompt: str) -> str:
        if self.provider == "gpt":
            return self._query_gpt(prompt)
        elif self.provider == "gemini":
            return self._query_gemini(prompt)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _query_gpt(self, prompt: str) -> str:
        openai.api_key = self.api_key
        response = openai.ChatCompletion.create(
            model=self.model or "gpt-4.1",
            messages=[
                {"role": "system", "content": "You are a helpful medical assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message["content"].strip()

    def _query_gemini(self, prompt: str) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model or 'gemini-2.0-flash'}:generateContent"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}]
        }
        response = requests.post(
            url + f"?key={self.api_key}",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]

    def _extract_classes_from_desc(self, desc_text: str):
        desc_text_lower = desc_text.lower()
        matched = []

        for fullname in self.all_fullnames:
            if fullname in desc_text_lower:
                matched.append(self.fullname_to_abbr[fullname])

        return list(set(matched)) if matched else ["unknown"]
    
    def match_instruction_to_descs(self, instruction: str, desc_dir: str, output_prefix: str):
        # Step 1: Load all descriptions
        desc_dict = {}
        for fname in os.listdir(desc_dir):
            if fname.endswith(".txt"):
                struct_id = fname.replace(".txt", "")
                with open(os.path.join(desc_dir, fname), "r") as f:
                    desc_dict[struct_id] = f.read().strip()

        # Step 2: Construct a single prompt
        description_lines = "\n".join(
            f"- {sid}: {desc}" for sid, desc in desc_dict.items()
        )
        prompt = (
            f"Instruction: {instruction}\n\n"
            f"Descriptions:\n{description_lines}\n\n"
            "Which anatomical structure best matches the instruction above?\n"
            "Answer with only the structure."
        )

        try:
            response = self.query(prompt)
        except Exception as e:
            response = f"[Error] {e}"

        best_match_id = response.strip()
        matched_desc = desc_dict.get(best_match_id, "")
        matched_class = self._extract_classes_from_desc(matched_desc)
        matched_class = matched_class[0] if matched_class else "unknown"

        # Output files
        with open(f"Outputs/{query_folder}/seg_target_cat_{output_prefix}.txt", "w") as f:
            f.write(best_match_id + "\n")

        with open(f"Outputs/{query_folder}/seg_target_res_{output_prefix}.json", "w") as f:
            json.dump({
                "instruction": instruction,
                "top_prediction": {
                    "structure": best_match_id,
                    "response": matched_desc,
                    "matched_category": matched_class
                },
                "all_descriptions": desc_dict
            }, f, indent=2)

        return (best_match_id, matched_desc, matched_class), desc_dict

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

    test_cases = [
        "please segment the structure of the metacarpal from the video",
        "please segment the small and white curve from the video",
        "segment the structures with hyperechoic features"
    ]

    parser = argparse.ArgumentParser(description="Test LLM instruction matching to descriptions")
    parser.add_argument("--desc_dir", type=str, required=True, help="Path to the folder containing description files")
    parser.add_argument("--instruction", type=str, required=True, help="Path to text or JSON file containing test instructions")
    parser.add_argument("--api_key", type=str, required=False, help="API key for the LLM")
    parser.add_argument("--model_name", type=str, default=None, help="Model name for the LLM (e.g., gpt-4, gemini-pro)")
    parser.add_argument("--folder", type=str, default=None, help="Query folder name")

    args = parser.parse_args()

    test_cases = load_test_cases(args.instruction)
    query_folder = args.folder
    gpt_router = LLMRouter(provider="gpt", api_key=gpt_api_key, model=args.model_name)
    gemini_router = LLMRouter(provider="gemini", api_key=gemini_api_key, model=args.model_name)
    for instruction in test_cases:
        if args.model_name.find("gpt") != -1:
            print(f"\n🧠 GPT Matching for: {instruction}")
            top, _ = gpt_router.match_instruction_to_descs(
                #model=args.model_name.strip(),
                instruction=instruction,
                desc_dir=args.desc_dir,
                output_prefix="gpt"
            )
            print(f"Top prediction: {top[0]} → Category: {top[1]}")
        elif args.model_name.find("gemini") != -1: 
            print(f"\n🌟 Gemini Matching for: {instruction}")
            top, _ = gemini_router.match_instruction_to_descs(
                instruction=instruction,
                #model=args.model_name.strip(),
                desc_dir=args.desc_dir,
                output_prefix="gemini"
            )
            print(f"Top prediction: {top[0]} → Category: {top[1]}")
