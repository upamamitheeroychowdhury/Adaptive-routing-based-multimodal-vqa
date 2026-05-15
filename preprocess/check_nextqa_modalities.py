import os
import json
import pickle
import torch

QA_JSON = "data/nextqa/nextqa_train_processed.json"
SIGLIP_PKL = "data/nextqa/nextqa_siglip_feats.pkl"
VCPT_PKL = "data/nextqa/yolo_vcpt/nextqa_vcpt_yolo.pkl"
ASR_JSONL = "data/nextqa/whisper_subtitles/nextqa_preprocessed_subtitles.jsonl"
FRAMES_ROOT = "data/nextqa/frames"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl_subtitles(path):
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            vid = str(item["vid_name"]).strip()
            data[vid] = item.get("sub", [])
    return data


def get_video_id(item):
    for key in ["video_id", "vid_name", "vid", "video"]:
        if key in item:
            return str(item[key]).strip()
    return None


def is_zero_feature(feat):
    if isinstance(feat, torch.Tensor):
        return feat.sum().item() == 0
    return False


def build_frame_keys(root):
    """
    Finds real frame folders by walking recursively.

    This works even if frames are stored like:
    data/nextqa/frames/0009/2574374895/frame_000.jpg

    or:
    data/nextqa/frames/2574374895/frame_000.jpg
    """
    frame_keys = set()

    if not os.path.isdir(root):
        return frame_keys

    for dirpath, _, files in os.walk(root):
        imgs = [
            f for f in files
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if len(imgs) == 0:
            continue

        folder_name = os.path.basename(dirpath)
        frame_keys.add(folder_name)

    return frame_keys


def main():
    qa = load_json(QA_JSON)
    qa_items = list(qa.values()) if isinstance(qa, dict) else qa

    qa_vids = set()
    for item in qa_items:
        vid = get_video_id(item)
        if vid:
            qa_vids.add(vid)

    print("Total QA samples:", len(qa_items))
    print("Unique QA videos:", len(qa_vids))

    with open(SIGLIP_PKL, "rb") as f:
        siglip = pickle.load(f)

    with open(VCPT_PKL, "rb") as f:
        vcpt = pickle.load(f)

    asr = load_jsonl_subtitles(ASR_JSONL)

    siglip_keys = set(map(str, siglip.keys()))
    vcpt_keys = set(map(str, vcpt.keys()))
    asr_keys = set(map(str, asr.keys()))
    frame_keys = build_frame_keys(FRAMES_ROOT)

    missing_siglip = sorted(qa_vids - siglip_keys)
    missing_vcpt = sorted(qa_vids - vcpt_keys)
    missing_asr = sorted(qa_vids - asr_keys)
    missing_frames = sorted(qa_vids - frame_keys)

    zero_siglip = [
        vid for vid in qa_vids
        if vid in siglip and is_zero_feature(siglip[vid])
    ]

    empty_vcpt = []
    for vid in qa_vids:
        if vid in vcpt:
            v = vcpt[vid]
            if v == {} or v == [] or v == "":
                empty_vcpt.append(vid)

    empty_asr = []
    for vid in qa_vids:
        if vid in asr:
            sub = asr[vid]
            if len(sub) == 0:
                empty_asr.append(vid)

    print("\n===== MISSING REPORT =====")
    print("Missing SigLIP:", len(missing_siglip))
    print("Zero SigLIP:", len(zero_siglip))
    print("Missing VCPT:", len(missing_vcpt))
    print("Empty VCPT:", len(empty_vcpt))
    print("Missing ASR:", len(missing_asr))
    print("Empty ASR:", len(empty_asr))
    print("Frame folders found:", len(frame_keys))
    print("Missing frames folder:", len(missing_frames))

    print("\nSample missing SigLIP:", missing_siglip[:10])
    print("Sample zero SigLIP:", zero_siglip[:10])
    print("Sample missing VCPT:", missing_vcpt[:10])
    print("Sample empty VCPT:", empty_vcpt[:10])
    print("Sample missing ASR:", missing_asr[:10])
    print("Sample empty ASR:", empty_asr[:10])
    print("Sample missing frames:", missing_frames[:10])


if __name__ == "__main__":
    main()