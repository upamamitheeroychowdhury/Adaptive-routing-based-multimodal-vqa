import json
import os
import pickle

IN_JSON = "data/nextqa/nextqa_train.json"

ASR_JSONL = "data/nextqa/whisper_subtitles/nextqa_preprocessed_subtitles.jsonl"
VCPT_PKL = "data/nextqa/yolo_vcpt/nextqa_vcpt_yolo.pkl"

OUT_JSON = "data/nextqa/nextqa_train_processed.json"


def load_subtitles(path):
    sub_dict = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            item = json.loads(line)
            vid = str(item["vid_name"]).strip()
            sub = item.get("sub", [])
            sub_text = " ".join([s.get("text", "") for s in sub]).strip()

            sub_dict[vid] = {
                "sub": sub,
                "sub_text": sub_text
            }

    return sub_dict


def vcpt_to_text(vcpt_item):
    if vcpt_item is None:
        return ""

    if isinstance(vcpt_item, dict):
        parts = []
        for k in sorted(vcpt_item.keys()):
            v = vcpt_item[k]
            if isinstance(v, str) and v.strip():
                parts.append(v.strip())
            elif isinstance(v, list):
                parts.extend([str(x) for x in v if str(x).strip()])
        return " ".join(parts).strip()

    if isinstance(vcpt_item, list):
        return " ".join([str(x) for x in vcpt_item if str(x).strip()]).strip()

    if isinstance(vcpt_item, str):
        return vcpt_item.strip()

    return ""


def normalize_vcpt(vcpt_item):
    """
    Keeps VCPT in TVQA-style frame-wise format:
    {"p0": "person, cup", "p1": "table", ...}
    """
    if isinstance(vcpt_item, dict):
        return {str(k): v for k, v in vcpt_item.items()}
    return {}


def main():
    with open(IN_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = list(data.values())

    subtitles = load_subtitles(ASR_JSONL)

    with open(VCPT_PKL, "rb") as f:
        vcpt_dict = pickle.load(f)

    converted = []
    missing_sub = 0
    missing_vcpt = 0

    for item in data:
        vid = str(item.get("vid_name", "")).strip()
        if not vid:
            continue

        sub_info = subtitles.get(vid, {"sub": [], "sub_text": ""})
        vcpt_item = vcpt_dict.get(vid, {})

        sub = sub_info["sub"]
        sub_text = sub_info["sub_text"]

        vcpt_framewise = normalize_vcpt(vcpt_item)
        vcpt_text = vcpt_to_text(vcpt_item)

        if not sub_text:
            missing_sub += 1

        if not vcpt_text:
            missing_vcpt += 1

        entry = {
            "qid": item.get("qid", ""),
            "vid_name": vid,

            "q": item.get("q", ""),
            "a0": item.get("a0", ""),
            "a1": item.get("a1", ""),
            "a2": item.get("a2", ""),
            "a3": item.get("a3", ""),
            "a4": item.get("a4", ""),

            "answer_idx": int(item.get("answer_idx", -1)),

            # subtitle fields
            "sub": sub_text,
            "sub_text": sub_text,
            "sub_segments": sub,
            "located_sub_text": sub_text,

            # VCPT fields
            "vcpt": vcpt_text,
            "vcpt_text": vcpt_text,
            "vcpt_framewise": vcpt_framewise,

            # compatibility fields
            "ts": item.get("ts", [0.0, 0.0]),
            "located_frame": item.get("located_frame", [0, 1]),
            "dataset": "nextqa"
        }

        converted.append(entry)

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(converted, f, indent=2, ensure_ascii=False)

    print("Saved:", OUT_JSON)
    print("Total converted:", len(converted))
    print("Missing/empty subtitle:", missing_sub)
    print("Missing/empty VCPT:", missing_vcpt)


if __name__ == "__main__":
    main()