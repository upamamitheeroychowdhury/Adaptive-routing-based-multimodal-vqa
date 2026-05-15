import os
import sys
import json
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vqa_datasets.unified_vqa_dataset import UnifiedVQADataset
from model.routed_multimodal_vqa import RoutedMultimodalVQA


# =========================================================
# CONFIG
# =========================================================

#CHECKPOINT_PATH = "results/routed_tvqa_star_moddrop_router/best.pt"
# CHECKPOINT_PATH = "results/routed_tvqa_star_zeroshot_regularized_run2/best.pt"
# OUTPUT_DIR = "results/routed_tvqa_star_zeroshot_regularized_run2/nextqa_eval"
CHECKPOINT_PATH = "results/routed_tvqa_star_zeroshot_regularized_run3/best.pt"
OUTPUT_DIR = "results/routed_tvqa_star_zeroshot_regularized_run3/nextqa_eval"
# CHECKPOINT_PATH = "results/routed_tvqa_star_zeroshot_strongdrop/best.pt"
# OUTPUT_DIR = "results/routed_tvqa_star_zeroshot_strongdrop/nextqa_eval"

NEXTQA_JSON = "data/nextqa/nextqa_test_processed.json"
NEXTQA_SIGLIP = "data/nextqa/nextqa_siglip_feats.pkl"
NEXTQA_VCPT = "data/nextqa/yolo_vcpt/nextqa_vcpt_yolo.pkl"

#OUTPUT_DIR = "results/routed_tvqa_star_moddrop_router/nextqa_eval"

BATCH_SIZE = 2
SIGLIP_DIM = 768
MAX_FRAMES = 32
MAX_OBJECTS_PER_FRAME = 10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# UTILS
# =========================================================

def move_to_device(batch, device):
    out = {}

    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v

    return out


def save_confusion_matrix(y_true, y_pred, output_dir):
    labels = [0, 1, 2, 3, 4]

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    cm_csv_path = os.path.join(output_dir, "confusion_matrix.csv")

    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{i}" for i in labels],
        columns=[f"pred_{i}" for i in labels]
    )

    cm_df.to_csv(cm_csv_path)

    print("\nConfusion Matrix:")
    print(cm_df)
    print("Saved confusion matrix CSV to:", cm_csv_path)

    plt.figure(figsize=(7, 6))
    plt.imshow(cm)
    plt.title("NExT-QA Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(labels)
    plt.yticks(labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center"
            )

    plt.colorbar()
    plt.tight_layout()

    cm_img_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_img_path, dpi=300)
    plt.close()

    print("Saved confusion matrix image to:", cm_img_path)

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
        output_dict=True
    )

    report_path = os.path.join(output_dir, "classification_report.json")

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Saved classification report to:", report_path)


# =========================================================
# TEST
# =========================================================

def test():
    print("Using device:", DEVICE)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -------------------------
    # Load checkpoint
    # -------------------------
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

    print("Loaded checkpoint:", CHECKPOINT_PATH)

    object_vocab = ckpt["object_vocab"]
    model_args = ckpt["args"]

    print("Checkpoint epoch:", ckpt.get("epoch", "unknown"))
    print("Checkpoint val_acc:", ckpt.get("val_acc", "unknown"))
    print("Object vocab size:", len(object_vocab))

    # -------------------------
    # Dataset
    # -------------------------
    dataset = UnifiedVQADataset(
        json_path=NEXTQA_JSON,
        siglip_pkl_path=NEXTQA_SIGLIP,
        vcpt_pkl_path=NEXTQA_VCPT,
        object_vocab=object_vocab,
        max_frames=MAX_FRAMES,
        max_objects_per_frame=MAX_OBJECTS_PER_FRAME,
        siglip_dim=SIGLIP_DIM,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    print("Dataset size:", len(dataset))

    # -------------------------
    # Model
    # -------------------------
    model = RoutedMultimodalVQA(
        num_objects=len(object_vocab),
        bert_name=model_args.get("bert_name", "bert-base-uncased"),
        siglip_dim=model_args.get("siglip_dim", SIGLIP_DIM),
        hidden_dim=768,
        dropout=model_args.get("dropout", 0.3),
        freeze_bert=model_args.get("freeze_bert", False),
    ).to(DEVICE)

    model.load_state_dict(ckpt["model"])
    model.eval()

    # -------------------------
    # Inference
    # -------------------------
    predictions = []

    total = 0
    correct = 0

    all_true = []
    all_pred = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing NExT-QA"):
            batch = move_to_device(batch, DEVICE)

            output = model(
                qa_input_ids=batch["qa_input_ids"],
                qa_attention_mask=batch["qa_attention_mask"],
                sub_input_ids=batch["sub_input_ids"],
                sub_attention_mask=batch["sub_attention_mask"],
                video_feats=batch["video_feats"],
                video_mask=batch["video_mask"],
                vcpt_obj_ids=batch["vcpt_obj_ids"],
                vcpt_obj_mask=batch["vcpt_obj_mask"],
            )

            scores = output["scores"]  # [B, 5]
            preds = scores.argmax(dim=1)

            labels = batch["label"]
            valid = labels >= 0

            if valid.sum().item() > 0:
                batch_correct = (preds[valid] == labels[valid]).sum().item()
                batch_total = valid.sum().item()

                correct += batch_correct
                total += batch_total

                all_true.extend(labels[valid].detach().cpu().tolist())
                all_pred.extend(preds[valid].detach().cpu().tolist())

            router_weights = output["router_weights"]  # [B, 5, 3]

            qids = batch["qid"]
            vid_names = batch["vid_name"]

            scores_cpu = scores.detach().cpu()
            preds_cpu = preds.detach().cpu()
            labels_cpu = labels.detach().cpu()
            router_cpu = router_weights.detach().cpu()

            for i in range(scores_cpu.size(0)):
                label_i = int(labels_cpu[i].item())
                pred_i = int(preds_cpu[i].item())

                is_correct = -1
                if label_i >= 0:
                    is_correct = int(pred_i == label_i)

                # router averaged over 5 answer options
                router_avg = router_cpu[i].mean(dim=0).tolist()

                predictions.append(
                    {
                        "qid": qids[i],
                        "vid_name": vid_names[i],
                        "prediction": pred_i,
                        "answer_idx": label_i,
                        "correct": is_correct,

                        "score_a0": float(scores_cpu[i, 0].item()),
                        "score_a1": float(scores_cpu[i, 1].item()),
                        "score_a2": float(scores_cpu[i, 2].item()),
                        "score_a3": float(scores_cpu[i, 3].item()),
                        "score_a4": float(scores_cpu[i, 4].item()),

                        "router_video": float(router_avg[0]),
                        "router_subtitle": float(router_avg[1]),
                        "router_vcpt": float(router_avg[2]),
                    }
                )

    # -------------------------
    # Accuracy
    # -------------------------
    if total > 0:
        acc = correct / total
        print("\nTotal evaluated:", total)
        print("Correct:", correct)
        print(f"NExT-QA Accuracy: {acc:.4f}")
    else:
        acc = None
        print("\nNo valid labels found. Saved predictions only.")

    # -------------------------
    # Save JSON
    # -------------------------
    json_path = os.path.join(OUTPUT_DIR, "nextqa_predictions.json")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "accuracy": acc,
                "total": total,
                "correct": correct,
                "checkpoint": CHECKPOINT_PATH,
                "results": predictions,
            },
            f,
            indent=2,
        )

    print("Saved JSON predictions to:", json_path)

    # -------------------------
    # Save CSV
    # -------------------------
    csv_path = os.path.join(OUTPUT_DIR, "nextqa_predictions.csv")

    df = pd.DataFrame(predictions)
    df.to_csv(csv_path, index=False)

    print("Saved CSV predictions to:", csv_path)

    # -------------------------
    # Confusion matrix
    # -------------------------
    if total > 0:
        save_confusion_matrix(
            y_true=all_true,
            y_pred=all_pred,
            output_dir=OUTPUT_DIR,
        )


if __name__ == "__main__":
    test()