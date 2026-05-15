import os
import cv2
from tqdm import tqdm

VIDEO_ROOT = "/media/twelvetb/work/projects/my_project/STAR/Charades_v1_480"
OUT_ROOT = "/media/twelvetb/work/projects/my_project/URC_PROJECT/data/star/frames"
NUM_FRAMES = 32

os.makedirs(OUT_ROOT, exist_ok=True)


def extract_frames(video_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total <= 0:
        cap.release()
        return

    frame_ids = [int(i * total / NUM_FRAMES) for i in range(NUM_FRAMES)]

    for i, frame_id in enumerate(frame_ids):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = cap.read()

        if ok:
            out_path = os.path.join(out_dir, f"frame_{i:03d}.jpg")
            cv2.imwrite(out_path, frame)

    cap.release()


def main():
    video_files = sorted([
        f for f in os.listdir(VIDEO_ROOT)
        if f.lower().endswith((".mp4", ".avi", ".mkv", ".webm"))
    ])

    print(f"Found {len(video_files)} videos.")

    for video_file in tqdm(video_files):
        video_path = os.path.join(VIDEO_ROOT, video_file)
        vid = os.path.splitext(video_file)[0]

        out_dir = os.path.join(OUT_ROOT, vid)

        # skip already extracted videos
        if os.path.isdir(out_dir) and len(os.listdir(out_dir)) >= NUM_FRAMES:
            continue

        extract_frames(video_path, out_dir)

    print("Done extracting STAR frames.")


if __name__ == "__main__":
    main()