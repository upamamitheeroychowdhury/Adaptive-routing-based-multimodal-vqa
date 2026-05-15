import json
import os

IN_FILES = {
    "train": "data/star/STAR_train.json",
    "val": "data/star/STAR_val.json",
    "test": "data/star/STAR_test.json",
}

OUT_FILES = {
    "train": "data/star/star_train_processed.json",
    "val": "data/star/star_val_processed.json",
    "test": "data/star/star_test_processed.json",
}


def convert_file(in_path, out_path):
    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    converted = []
    skipped = 0

    for item in data:
        vid = item.get("video_id", "")
        q = item.get("question", "")
        answer_text = item.get("answer", None)

        choices = [c["choice"] for c in item.get("choices", [])]

        if len(choices) < 4:
            skipped += 1
            continue

        if answer_text is None:
            answer_idx = -1   # for STAR_test.json
        else:
            try:
                answer_idx = choices.index(answer_text)
            except ValueError:
                skipped += 1
                continue

        entry = {
            "qid": item.get("question_id", ""),
            "vid_name": vid,
            "q": q,
            "a0": choices[0],
            "a1": choices[1],
            "a2": choices[2],
            "a3": choices[3],
            "a4": "",                 # dummy answer for TVQA-style compatibility
            "answer_idx": answer_idx,
            "ts": [float(item.get("start", 0.0)), float(item.get("end", 0.0))],
            "dataset": "star"
        }

        converted.append(entry)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(converted, f, indent=2, ensure_ascii=False)

    print("Saved:", out_path)
    print("Total:", len(converted))
    print("Skipped:", skipped)


def main():
    for split in IN_FILES:
        convert_file(IN_FILES[split], OUT_FILES[split])


if __name__ == "__main__":
    main()