import json
import pickle
import os

IN_JSON = "data/nextqa/yolo_vcpt/nextqa_vcpt_yolo.json"
OUT_PKL = "data/nextqa/yolo_vcpt/nextqa_vcpt_yolo.pkl"

os.makedirs(os.path.dirname(OUT_PKL), exist_ok=True)

data = json.load(open(IN_JSON, "r", encoding="utf-8"))
data = {str(k): v for k, v in data.items()}

with open(OUT_PKL, "wb") as f:
    pickle.dump(data, f)

print("Saved:", OUT_PKL)
print("Total:", len(data))