import os
import json
import pickle
import cv2
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel

os.environ["HF_HOME"] = "/media/twelvetb/hf_cache"
os.environ["TORCH_HOME"] = "/media/twelvetb/torch_cache"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "google/siglip-base-patch16-224"

NEXTQA_JSONS = [
    "data/nextqa/nextqa_train.json",
]

VIDEO_ROOT = "data/nextqa/videos"
OUT_PATH = "data/nextqa/nextqa_siglip_feats.pkl"
MAX_FRAMES = 8


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


def load_video_frames(video_path, max_frames=8):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_ids = sample_frame_indices(total_frames, max_frames)

    frames = []

    for fid in frame_ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ok, frame = cap.read()

        if not ok or frame is None:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))

    cap.release()
    return frames


@torch.no_grad()
def encode_images(images, processor, model):
    if len(images) == 0:
        return torch.zeros(MAX_FRAMES, 768)

    inputs = processor(images=images, return_tensors="pt").to(DEVICE)
    outputs = model.vision_model(**inputs)

    feats = outputs.pooler_output.detach().cpu()

    if feats.shape[0] < MAX_FRAMES:
        pad = torch.zeros(MAX_FRAMES - feats.shape[0], feats.shape[1])
        feats = torch.cat([feats, pad], dim=0)

    return feats[:MAX_FRAMES]


def main():
    needed_vids = load_needed_vids()
    print("Needed NExT-QA videos:", len(needed_vids))

    video_index = build_video_index()

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()

    final_feats = {}

    if os.path.exists(OUT_PATH):
        with open(OUT_PATH, "rb") as f:
            final_feats = pickle.load(f)
        print("Loaded existing:", len(final_feats))

    for vid in tqdm(sorted(needed_vids)):
        if vid in final_feats:
            continue

        video_path = video_index.get(vid)

        if video_path is None:
            print("Missing video for:", vid)
            final_feats[vid] = torch.zeros(MAX_FRAMES, 768)
            continue

        frames = load_video_frames(video_path, MAX_FRAMES)
        feats = encode_images(frames, processor, model)
        final_feats[vid] = feats

        if len(final_feats) % 200 == 0:
            with open(OUT_PATH, "wb") as f:
                pickle.dump(final_feats, f)

    with open(OUT_PATH, "wb") as f:
        pickle.dump(final_feats, f)

    missing = sum(1 for v in final_feats.values() if isinstance(v, torch.Tensor) and v.sum().item() == 0)

    print("Saved:", OUT_PATH)
    print("Total saved:", len(final_feats))
    print("Missing NExT-QA features:", missing)


if __name__ == "__main__":
    main()