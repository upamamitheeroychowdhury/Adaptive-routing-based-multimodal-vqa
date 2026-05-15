import os
import json
import cv2

os.environ["YOLO_CONFIG_DIR"] = "/media/twelvetb/ultralytics_cache"

from ultralytics import YOLO
from tqdm import tqdm

VIDEO_ROOT = "/media/twelvetb/work/projects/my_project/STAR/Charades_v1_480"
OUTPUT_JSON = "data/star/vcpt/star_vcpt_yolo.json"

NUM_SAMPLED_FRAMES = 16
CONF_THRES = 0.25

os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)

model = YOLO("yolov8n.pt")


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

        # TVQA-style: each sampled frame has a string of visual concepts
        vcpt[f"p{i}"] = ", ".join(labels)

    cap.release()
    return vcpt


def find_videos(root):
    videos = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith((".mp4", ".avi", ".mkv", ".webm")):
                videos.append(os.path.join(dirpath, f))
    return sorted(videos)


def main():
    videos = find_videos(VIDEO_ROOT)
    print(f"Found {len(videos)} STAR videos.")

    all_vcpt = {}

    if os.path.exists(OUTPUT_JSON):
        with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
            all_vcpt = json.load(f)
        print(f"Loaded existing VCPT for {len(all_vcpt)} videos.")

    for idx, video_path in enumerate(tqdm(videos), start=1):
        vid = os.path.splitext(os.path.basename(video_path))[0]

        if vid in all_vcpt:
            continue

        try:
            all_vcpt[vid] = extract_tvqa_style_vcpt(video_path)
        except Exception as e:
            print(f"Failed on {vid}: {e}")
            all_vcpt[vid] = {}

        if idx % 50 == 0:
            with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
                json.dump(all_vcpt, f, indent=2, ensure_ascii=False)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_vcpt, f, indent=2, ensure_ascii=False)

    print("Saved:", OUTPUT_JSON)


if __name__ == "__main__":
    main()