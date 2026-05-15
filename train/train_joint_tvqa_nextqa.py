import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

sys.path.insert(0, os.getcwd())

from vqa_datasets.unified_vqa_dataset import UnifiedVQADataset
from model.multimodal_vqa import MultimodalVQA


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TVQA_TRAIN_JSON = "data/tvqa/tvqa_train_processed.json"
TVQA_VAL_JSON = "data/tvqa/tvqa_val_processed.json"
TVQA_SIGLIP = "data/tvqa/tvqa_siglip_feats.pkl"

NEXTQA_TRAIN_JSON = "data/nextqa/nextqa_train.json"
NEXTQA_SIGLIP = "data/nextqa/nextqa_siglip_feats.pkl"

BATCH_SIZE = 4
EPOCHS = 5
LR = 2e-5

SAVE_PATH = "checkpoints/joint_tvqa_nextqa_multimodal.pt"

os.makedirs("checkpoints", exist_ok=True)


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for batch in tqdm(loader):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        frame_feats = batch["frame_feats"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        optimizer.zero_grad()

        scores = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            frame_feats=frame_feats
        )

        loss = criterion(scores, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = scores.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    for batch in tqdm(loader):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        frame_feats = batch["frame_feats"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        scores = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            frame_feats=frame_feats
        )

        loss = criterion(scores, labels)

        total_loss += loss.item()

        preds = scores.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


def main():
    print("Device:", DEVICE)

    tvqa_train = UnifiedVQADataset(
        json_path=TVQA_TRAIN_JSON,
        siglip_path=TVQA_SIGLIP
    )

    nextqa_train = UnifiedVQADataset(
        json_path=NEXTQA_TRAIN_JSON,
        siglip_path=NEXTQA_SIGLIP
    )

    tvqa_val = UnifiedVQADataset(
        json_path=TVQA_VAL_JSON,
        siglip_path=TVQA_SIGLIP
    )

    joint_train = ConcatDataset([tvqa_train, nextqa_train])

    print("TVQA train:", len(tvqa_train))
    print("NExT-QA train:", len(nextqa_train))
    print("Joint train:", len(joint_train))
    print("TVQA val:", len(tvqa_val))

    train_loader = DataLoader(
        joint_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )

    val_loader = DataLoader(
        tvqa_val,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )

    model = MultimodalVQA().to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion
        )

        val_loss, val_acc = evaluate(
            model,
            val_loader,
            criterion
        )

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train Acc:  {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")
        print(f"Val Acc:    {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print("Saved best model:", SAVE_PATH)

    print("Best TVQA validation accuracy:", best_val_acc)


if __name__ == "__main__":
    main()