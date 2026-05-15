import os
import cv2
from tqdm import tqdm

VIDEO_ROOT = "data/nextqa/videos"
OUT_ROOT = "data/nextqa/frames"
NUM_FRAMES = 32

os.makedirs(OUT_ROOT, exist_ok=True)


def sample_frame_indices(total_frames, num_samples):
    if total_frames <= 0:
        return []
    if total_frames <= num_samples:
        return list(range(total_frames))
    step = total_frames / float(num_samples)
    return [int(i * step) for i in range(num_samples)]


def find_video_file(folder):
    for f in os.listdir(folder):
        if f.lower().endswith((".mp4", ".avi", ".mkv", ".webm")):
            return os.path.join(folder, f)
    return None


def extract_frames(video_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Failed:", video_path)
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_ids = sample_frame_indices(total, NUM_FRAMES)

    for i, fid in enumerate(frame_ids):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ok, frame = cap.read()

        if ok and frame is not None:
            out_path = os.path.join(out_dir, f"frame_{i:03d}.jpg")
            cv2.imwrite(out_path, frame)

    cap.release()


def main():
    video_folders = sorted(os.listdir(VIDEO_ROOT))

    print("Total video folders:", len(video_folders))

    for vid in tqdm(video_folders):
        folder = os.path.join(VIDEO_ROOT, vid)

        if not os.path.isdir(folder):
            continue

        video_path = find_video_file(folder)
        if video_path is None:
            print("No video found in:", folder)
            continue

        out_dir = os.path.join(OUT_ROOT, vid)

        # skip if already extracted
        if os.path.exists(out_dir) and len(os.listdir(out_dir)) >= NUM_FRAMES:
            continue

        extract_frames(video_path, out_dir)

    print("Done extracting frames.")


if __name__ == "__main__":
    main()