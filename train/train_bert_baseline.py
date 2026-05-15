import os
import json
import torch
from torch.utils.data import DataLoader
from transformers import AdamW
import matplotlib.pyplot as plt
from tvqa_terra_dataset import TVQATerraDataset
from model.terra_vqa import TERRA_VQA


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_JSON = "data/tvqa_train_processed.json"
VALID_JSON = "data/tvqa_val_processed.json"
VCPT_PATH = "data/det_visual_concepts_hq.pickle"

SAVE_DIR = "results/terra_vqa"
MODEL_NAME = "bert-base-uncased"


BATCH_SIZE = 2
EPOCHS = 1
# BATCH_SIZE = 4
# EPOCHS = 5
LR = 2e-5
LAMBDA_REASON = 0.1


def evaluate(model, loader):
    model.eval()
    total = 0
    correct = 0
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in batch.items()}

            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                reasoning_input_ids=batch["reasoning_input_ids"],
                reasoning_attention_mask=batch["reasoning_attention_mask"],
                lambda_reason=LAMBDA_REASON
            )

            preds = out["scores"].argmax(dim=1)
            correct += (preds == batch["labels"]).sum().item()
            total += batch["labels"].size(0)
            total_loss += out["loss"].item() * batch["labels"].size(0)

    return total_loss / total, correct / total

def save_training_plots(history, save_dir):
    epochs = [x["epoch"] for x in history]

    train_loss = [x["train_loss"] for x in history]
    val_loss = [x["val_loss"] for x in history]

    train_acc = [x["train_acc"] for x in history]
    val_acc = [x["val_acc"] for x in history]

    # Loss curve
    plt.figure()
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("TERRA-VQA Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Accuracy curve
    plt.figure()
    plt.plot(epochs, train_acc, label="Train Accuracy")
    plt.plot(epochs, val_acc, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("TERRA-VQA Accuracy Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "accuracy_curve.png"), dpi=300, bbox_inches="tight")
    plt.close()


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    train_set = TVQATerraDataset(
        json_path=TRAIN_JSON,
        vcpt_path=VCPT_PATH,
        tokenizer_name=MODEL_NAME,
        with_ts=True
    )

    valid_set = TVQATerraDataset(
        json_path=VALID_JSON,
        vcpt_path=VCPT_PATH,
        tokenizer_name=MODEL_NAME,
        with_ts=True
    )

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False)

    model = TERRA_VQA(model_name=MODEL_NAME).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR)

    best_acc = 0.0
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0
        correct = 0
        total_loss = 0.0

        for batch in train_loader:
            batch = {k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in batch.items()}

            optimizer.zero_grad()

            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                reasoning_input_ids=batch["reasoning_input_ids"],
                reasoning_attention_mask=batch["reasoning_attention_mask"],
                lambda_reason=LAMBDA_REASON
            )

            loss = out["loss"]
            loss.backward()
            optimizer.step()

            preds = out["scores"].argmax(dim=1)
            correct += (preds == batch["labels"]).sum().item()
            total += batch["labels"].size(0)
            total_loss += loss.item() * batch["labels"].size(0)

        train_loss = total_loss / total
        train_acc = correct / total

        val_loss, val_acc = evaluate(model, valid_loader)

        print(
            f"Epoch {epoch} | "
            f"Train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"Val loss {val_loss:.4f} acc {val_acc:.4f}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_valid.pth"))
            print(f"Saved best model: {best_acc:.4f}")

        with open(os.path.join(SAVE_DIR, "history.json"), "w") as f:
            json.dump(history, f, indent=2)
        save_training_plots(history, SAVE_DIR)

    print("Best validation accuracy:", best_acc)


if __name__ == "__main__":
    main()