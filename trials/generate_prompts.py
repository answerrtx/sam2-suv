import json
from itertools import cycle

# Load provided data
with open("./suv_all.json", "r") as f:
    suv_data = json.load(f)

with open("./color_map_full.json", "r") as f:
    color_data = json.load(f)

# Reverse lookup for class type
label_to_class = {}
for cls, labels in suv_data.items():
    if cls != "color_map":
        for l in labels:
            label_to_class[l] = cls

# RGB & full name lookup
abbrev_to_rgb = {item["Abbreviation"]: tuple(item["RGB"]) for item in color_data}
abbrev_to_full = {item["Abbreviation"]: item["Full Name"] for item in color_data}

# Given list of label scenes
scenes = [
    ["Deltoid", "Hum", "Subscap"],
    ["Deltoid", "Hum", "SSP"],
    ["Deltoid", "Biceps", "Hum"],
    ["ACR", "CLV", "SSP"],
    ["Deltoid", "HH", "InfSp"],
    ["MC"],
    ["PrxPX", "UCL", "MC"],
    ["Rad", "FT", "LU", "Sca"],
    ["VA"],
    ["VA"],
    ["A", "FH", "IP", "FN", "AIIS"],
    ["GtrTroch"],
    ["P"],
    ["MCL", "Fem", "Tib"],
    ["Fem"],
]

# Generate prompts
def generate_prompt(scene):
    prompt_intro = "The first is an ultrasound image; the second is a mask. "
    mask_lines = []
    for label in scene:
        rgb = abbrev_to_rgb.get(label, ("?", "?", "?"))
        rgb_str = f"({rgb[0]}, {rgb[1]}, {rgb[2]})"
        full_name = abbrev_to_full.get(label, label)
        mask_lines.append(f"the mask of color {rgb_str} indicates the {full_name}")
    mask_part = "; ".join(mask_lines) + "."

    # Analysis prompt
    analysis = " Briefly describe its the transducer direction, anatomical class, color (not mask color), shape, and morphomimetics. Then summarize in a paragraph. When summrazing, don't include the mask color."

    return prompt_intro + mask_part + analysis

prompts = [generate_prompt(scene) for scene in scenes]

import pandas as pd

# Convert to DataFrame for display
df = pd.DataFrame({
    "Scene": [", ".join(scene) for scene in scenes],
    "Prompt": prompts
})

df.to_csv("prompts_output2.csv", index=False)
