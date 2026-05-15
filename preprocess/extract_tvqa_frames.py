import os
import tarfile
import zipfile
import shutil
from tqdm import tqdm

SRC_ROOT = "/media/twelvetb/work/projects/my_project/TVQA/frames_hq"
OUT_ROOT = "/media/twelvetb/work/projects/my_project/URC_PROJECT/data/tvqa/frames"
TEMP_ROOT = "/media/twelvetb/work/projects/my_project/URC_PROJECT/data/tvqa/temp_extract"

os.makedirs(OUT_ROOT, exist_ok=True)
os.makedirs(TEMP_ROOT, exist_ok=True)

def copy_images_as_star_style(temp_dir):
    for dirpath, _, files in os.walk(temp_dir):
        imgs = sorted([f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        if not imgs:
            continue

        clip_name = os.path.basename(dirpath)
        out_dir = os.path.join(OUT_ROOT, clip_name)
        os.makedirs(out_dir, exist_ok=True)

        for i, img in enumerate(imgs):
            src = os.path.join(dirpath, img)
            dst = os.path.join(out_dir, f"frame_{i:03d}.jpg")
            if not os.path.exists(dst):
                shutil.copy(src, dst)

def main():
    files = [
        os.path.join(SRC_ROOT, f)
        for f in os.listdir(SRC_ROOT)
        if f.startswith("tvqa_video_frames")
    ]

    print("Found files:", len(files))

    bad = []

    for fpath in tqdm(files):
        name = os.path.basename(fpath)
        temp_dir = os.path.join(TEMP_ROOT, name)
        shutil.rmtree(temp_dir, ignore_errors=True)
        os.makedirs(temp_dir, exist_ok=True)

        try:
            if zipfile.is_zipfile(fpath):
                with zipfile.ZipFile(fpath, "r") as z:
                    z.extractall(temp_dir)

            else:
                with tarfile.open(fpath, "r:*") as tar:
                    tar.extractall(temp_dir)

            copy_images_as_star_style(temp_dir)

        except Exception as e:
            print("Skipping bad archive:", name, "|", e)
            bad.append(name)

        shutil.rmtree(temp_dir, ignore_errors=True)

    print("Saved frames to:", OUT_ROOT)
    print("Bad archives:", len(bad))
    for b in bad:
        print(b)

if __name__ == "__main__":
    main()