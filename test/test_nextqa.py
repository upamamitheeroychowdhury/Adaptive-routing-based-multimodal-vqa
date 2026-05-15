import os
import json
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import BertTokenizerFast

from model.terra_vqa import TERRA_VQA


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NEXTQA_JSON = "data/nextqa_test_processed_with_sub_vcpt.json"
CHECKPOINT = "results/terra_vqa/best_valid.pth"
OUTPUT_JSON = "results/nextqa_terra_vqa_predictions.json"

MODEL_NAME = "bert-base-uncased"

# Use 1 for quick/debug test. Later change to 4 if GPU memory allows.
BATCH_SIZE = 1
MAX_LEN = 128

# Set to None for full test, or 200 for quick test.
MAX_TEST_SAMPLES = 200


class NextQATerraDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        q = item["q"]
        sub = item.get("located_sub_text", "")
        vcpt = item.get("vcpt", "")
        answers = [item[f"a{i}"] for i in range(5)]

        ids = []
        masks = []

        for ans in answers:
            text = f"question: {q} answer: {ans} subtitle: {sub} visual concepts: {vcpt}"

            enc = self.tokenizer(
                text,
                max_length=MAX_LEN,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

            ids.append(enc["input_ids"].squeeze(0))
            masks.append(enc["attention_mask"].squeeze(0))

        return {
            "input_ids": torch.stack(ids, dim=0),
            "attention_mask": torch.stack(masks, dim=0),
            "labels": torch.tensor(int(item["answer_idx"])).long(),
            "qid": item["qid"]
        }


def main():
    print("Using device:", DEVICE)

    dataset = NextQATerraDataset(NEXTQA_JSON)

    if MAX_TEST_SAMPLES is not None:
        n = min(MAX_TEST_SAMPLES, len(dataset))
        dataset = Subset(dataset, range(n))
        print(f"Quick test mode: using {n} samples")
    else:
        print(f"Full test mode: using {len(dataset)} samples")

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = TERRA_VQA(model_name=MODEL_NAME).to(DEVICE)
    checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model.eval()

    total = 0
    correct = 0
    results = []

    with torch.no_grad():
        for step, batch in enumerate(loader):
            if step % 20 == 0:
                print(f"Testing batch {step}/{len(loader)}")

            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            preds = out["scores"].argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            for qid, pred, gold in zip(batch["qid"], preds.cpu().tolist(), labels.cpu().tolist()):
                results.append({
                    "qid": int(qid),
                    "pred": int(pred),
                    "gold": int(gold)
                })

    acc = correct / total

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "accuracy": acc,
            "num_examples": total,
            "results": results
        }, f, indent=2)

    print(f"TERRA-VQA NExT-QA accuracy: {acc:.4f}")
    print(f"Saved predictions to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()