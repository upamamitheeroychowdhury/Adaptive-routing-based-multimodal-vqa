import json
import os

IN_JSON = "data/nextqa/whisper_subtitles/nextqa_asr.json"
OUT_JSONL = "data/nextqa/whisper_subtitles/nextqa_preprocessed_subtitles.jsonl"

os.makedirs(os.path.dirname(OUT_JSONL), exist_ok=True)

data = json.load(open(IN_JSON, "r", encoding="utf-8"))

with open(OUT_JSONL, "w", encoding="utf-8") as f:
    for vid, item in data.items():
        sub = []
        for seg in item.get("segments", []):
            text = seg.get("text", "").strip()
            if text:
                sub.append({
                    "text": text,
                    "start": float(seg.get("start", 0.0)),
                    "end": float(seg.get("end", 0.0))
                })

        f.write(json.dumps({
            "vid_name": str(vid),
            "sub": sub
        }, ensure_ascii=False) + "\n")

print("Saved:", OUT_JSONL)