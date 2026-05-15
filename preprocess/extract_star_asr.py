import os
import json
import whisper
from tqdm import tqdm

VIDEO_ROOT = "/media/twelvetb/work/projects/my_project/STAR/Charades_v1_480"
OUT_DIR = "data/star/subtitles"
OUT_JSONL = os.path.join(OUT_DIR, "star_preprocessed_subtitles.jsonl")

os.makedirs(OUT_DIR, exist_ok=True)

model = whisper.load_model("base")  # or "small"

def find_videos(root):
    videos = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith((".mp4", ".avi", ".mkv", ".webm")):
                videos.append(os.path.join(dirpath, f))
    return sorted(videos)

videos = find_videos(VIDEO_ROOT)
print(f"Found {len(videos)} STAR videos")

done = set()
if os.path.exists(OUT_JSONL):
    with open(OUT_JSONL, "r") as f:
        for line in f:
            try:
                done.add(json.loads(line)["vid_name"])
            except:
                pass

with open(OUT_JSONL, "a") as fout:
    for video_path in tqdm(videos):
        vid = os.path.splitext(os.path.basename(video_path))[0]

        if vid in done:
            continue

        try:
            result = model.transcribe(video_path, fp16=True)

            sub = []
            for seg in result.get("segments", []):
                text = seg["text"].strip()
                if text:
                    sub.append({
                        "text": text,
                        "start": float(seg["start"]),
                        "end": float(seg["end"])
                    })

            entry = {
                "vid_name": vid,
                "sub": sub
            }

            fout.write(json.dumps(entry) + "\n")
            fout.flush()

        except Exception as e:
            print(f"Error processing {vid}: {e}")

print("Saved:", OUT_JSONL)