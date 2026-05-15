import os
import json
import pickle
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel

os.environ["HF_HOME"] = "/media/twelvetb/hf_cache"
os.environ["TORCH_HOME"] = "/media/twelvetb/torch_cache"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "google/siglip-base-patch16-224"

TVQA_JSONS = [
    "data/tvqa/tvqa_train_processed.json",
    "data/tvqa/tvqa_val_processed.json",
]

FRAME_ROOT = "data/tvqa/frames/frames_hq"
OUT_PATH = "data/tvqa/tvqa_siglip_feats.pkl"
MAX_FRAMES = 8


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

def sample_indices(n, k):
    if n <= 0:
        return []
    if n <= k:
        return list(range(n))
    step = n / float(k)
    return [int(i * step) for i in range(k)]


def load_frames(frame_dir):
    if not os.path.isdir(frame_dir):
        return []

    imgs = sorted([
        os.path.join(frame_dir, f)
        for f in os.listdir(frame_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    idxs = sample_indices(len(imgs), MAX_FRAMES)

    frames = []
    for i in idxs:
        try:
            frames.append(Image.open(imgs[i]).convert("RGB"))
        except Exception:
            pass

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
    print("Needed TVQA clips:", len(needed_vids))

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()

    final_feats = {}

    if os.path.exists(OUT_PATH):
        with open(OUT_PATH, "rb") as f:
            final_feats = pickle.load(f)
        print("Loaded existing:", len(final_feats))

    missing = 0

    for vid in tqdm(sorted(needed_vids)):
        if vid in final_feats:
            continue

        frame_dir = vid_to_frame_dir(vid)
        frames = load_frames(frame_dir)

        if len(frames) == 0:
            missing += 1
            final_feats[vid] = torch.zeros(MAX_FRAMES, 768)
        else:
            final_feats[vid] = encode_images(frames, processor, model)

        if len(final_feats) % 500 == 0:
            with open(OUT_PATH, "wb") as f:
                pickle.dump(final_feats, f)

    with open(OUT_PATH, "wb") as f:
        pickle.dump(final_feats, f)

    zero_count = sum(1 for v in final_feats if final_feats[v].sum() == 0)

    print("Saved:", OUT_PATH)
    print("Total saved:", len(final_feats))
    print("Missing TVQA features:", zero_count)


if __name__ == "__main__":
    main()