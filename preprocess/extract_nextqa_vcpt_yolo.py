import os
import json
import pickle
import cv2

os.environ["YOLO_CONFIG_DIR"] = "/media/twelvetb/ultralytics_cache"

from ultralytics import YOLO
from tqdm import tqdm

VIDEO_ROOT = "data/nextqa/videos"
OUTPUT_JSON = "data/nextqa/yolo_vcpt/nextqa_vcpt_yolo.json"
OUTPUT_PKL = "data/nextqa/yolo_vcpt/nextqa_vcpt_yolo.pkl"

NEXTQA_JSONS = [
    "data/nextqa/nextqa_train.json",
]

NUM_SAMPLED_FRAMES = 16
CONF_THRES = 0.25

os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)

model = YOLO("yolov8n.pt")


def load_needed_vids():
    needed = set()

    for path in NEXTQA_JSONS:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            data = list(data.values())

        for item in data:
            vid = str(item.get("vid_name", "")).strip()
            if vid:
                needed.add(vid)

    return needed


def build_video_index():
    video_index = {}

    for root, _, files in os.walk(VIDEO_ROOT):
        for f in files:
            if f.lower().endswith((".mp4", ".avi", ".mkv", ".webm")):
                vid_name = os.path.splitext(f)[0]
                video_index[vid_name] = os.path.join(root, f)

    print("Indexed videos:", len(video_index))
    return video_index


def sample_frame_indices(total_frames, num_samples):
    if total_frames <= 0:
        return []
    if total_frames <= num_samples:
        return list(range(total_frames))

    step = total_frames / float(num_samples)
    return [int(i * step) for i in range(num_samples)]


def unique_preserve_order(items):
    seen = set()
    out = []

    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)

    return out


def extract_tvqa_style_vcpt(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return {}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_ids = sample_frame_indices(total_frames, NUM_SAMPLED_FRAMES)

    vcpt = {}

    for i, fid in enumerate(frame_ids):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ok, frame = cap.read()

        if not ok or frame is None:
            vcpt[f"p{i}"] = ""
            continue

        results = model(frame, verbose=False, conf=CONF_THRES)

        labels = []

        for r in results:
            if r.boxes is None:
                continue

            for cls_id in r.boxes.cls.tolist():
                labels.append(model.names[int(cls_id)])

        labels = unique_preserve_order(labels)
        vcpt[f"p{i}"] = ", ".join(labels)

    cap.release()
    return vcpt


def main():
    needed_vids = load_needed_vids()
    print("Needed NExT-QA videos:", len(needed_vids))

    video_index = build_video_index()

    all_vcpt = {}

    if os.path.exists(OUTPUT_JSON):
        with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
            all_vcpt = json.load(f)
        print("Loaded existing VCPT:", len(all_vcpt))

    for idx, vid in enumerate(tqdm(sorted(needed_vids)), start=1):
        if vid in all_vcpt:
            continue

        video_path = video_index.get(vid)

        if video_path is None:
            print("Missing video for:", vid)
            all_vcpt[vid] = {}
            continue

        try:
            all_vcpt[vid] = extract_tvqa_style_vcpt(video_path)
        except Exception as e:
            print(f"Failed on {vid}: {e}")
            all_vcpt[vid] = {}

        if idx % 200 == 0:
            with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
                json.dump(all_vcpt, f, indent=2, ensure_ascii=False)

            with open(OUTPUT_PKL, "wb") as f:
                pickle.dump(all_vcpt, f)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_vcpt, f, indent=2, ensure_ascii=False)

    with open(OUTPUT_PKL, "wb") as f:
        pickle.dump(all_vcpt, f)

    missing = sum(1 for v in all_vcpt.values() if len(v) == 0)

    print("Saved JSON:", OUTPUT_JSON)
    print("Saved PKL:", OUTPUT_PKL)
    print("Total saved:", len(all_vcpt))
    print("Missing/empty VCPT:", missing)


if __name__ == "__main__":
    main()