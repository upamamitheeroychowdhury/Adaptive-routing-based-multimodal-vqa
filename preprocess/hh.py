import json
import pickle

JSON_PATH = "data/star/vcpt/star_vcpt_yolo_tvqa_style.json"
PKL_PATH = "data/star/vcpt/star_vcpt_yolo_tvqa_style.pkl"

with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# (optional but recommended) ensure keys are strings
data = {str(k): v for k, v in data.items()}

with open(PKL_PATH, "wb") as f:
    pickle.dump(data, f)

print("Saved:", PKL_PATH)