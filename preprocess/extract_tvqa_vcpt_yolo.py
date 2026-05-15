import os
import json
import pickle
from PIL import Image

os.environ["YOLO_CONFIG_DIR"] = "/media/twelvetb/ultralytics_cache"

from ultralytics import YOLO
from tqdm import tqdm

FRAME_ROOT = "data/tvqa/frames/frames_hq"
OUTPUT_JSON = "data/tvqa/vcpt/tvqa_vcpt_yolo.json"
OUTPUT_PKL = "data/tvqa/vcpt/tvqa_vcpt_yolo.pkl"

TVQA_JSONS = [
    "data/tvqa/tvqa_train_processed.json",
    "data/tvqa/tvqa_val_processed.json",
]

NUM_SAMPLED_FRAMES = 16
CONF_THRES = 0.25

os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)

model = YOLO("yolov8n.pt")


def load_needed_vids():
    needed = set()

    for path in TVQA_JSONS:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            data = list(data.values())

        for item in data:
            vid = str(item.get("vid_name", "")).strip()
            if vid:
                needed.add(vid)

    return needed


def vid_to_frame_dir(vid):
    show = vid.split("_", 1)[0]
    return os.path.join(FRAME_ROOT, f"{show}_frames", vid)


def sample_indices(total, num_samples):
    if total <= 0:
        return []
    if total <= num_samples:
        return list(range(total))

    step = total / float(num_samples)
    return [int(i * step) for i in range(num_samples)]


def unique_preserve_order(items):
    seen = set()
    out = []

    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)

    return out


def extract_tvqa_style_vcpt_from_frames(frame_dir):
    if not os.path.isdir(frame_dir):
        return {}

    image_paths = sorted([
        os.path.join(frame_dir, f)
        for f in os.listdir(frame_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    frame_ids = sample_indices(len(image_paths), NUM_SAMPLED_FRAMES)

    vcpt = {}

    for i, idx in enumerate(frame_ids):
        img_path = image_paths[idx]

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            vcpt[f"p{i}"] = ""
            continue

        results = model(img, verbose=False, conf=CONF_THRES)

        labels = []

        for r in results:
            if r.boxes is None:
                continue

            for cls_id in r.boxes.cls.tolist():
                labels.append(model.names[int(cls_id)])

        labels = unique_preserve_order(labels)

        vcpt[f"p{i}"] = ", ".join(labels)

    return vcpt


def main():
    needed_vids = load_needed_vids()
    print("Needed TVQA clips:", len(needed_vids))

    all_vcpt = {}

    if os.path.exists(OUTPUT_JSON):
        with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
            all_vcpt = json.load(f)
        print("Loaded existing VCPT:", len(all_vcpt))

    for idx, vid in enumerate(tqdm(sorted(needed_vids)), start=1):
        if vid in all_vcpt:
            continue

        frame_dir = vid_to_frame_dir(vid)

        try:
            all_vcpt[vid] = extract_tvqa_style_vcpt_from_frames(frame_dir)
        except Exception as e:
            print(f"Failed on {vid}: {e}")
            all_vcpt[vid] = {}

        if idx % 500 == 0:
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