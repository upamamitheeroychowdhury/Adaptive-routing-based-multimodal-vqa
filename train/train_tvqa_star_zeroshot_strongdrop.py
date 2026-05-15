import os
import sys
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vqa_datasets.unified_vqa_dataset import UnifiedVQADataset, build_object_vocab
from model.routed_multimodal_vqa import RoutedMultimodalVQA


def move_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def freeze_lower_bert_layers(model, freeze_until_layer=5):
    """
    Freeze BERT embeddings + lower encoder layers.
    Keeps upper BERT layers trainable.
    """
    for name, param in model.bert.named_parameters():
        if name.startswith("embeddings"):
            param.requires_grad = False

        for layer_id in range(freeze_until_layer + 1):
            if name.startswith(f"encoder.layer.{layer_id}."):
                param.requires_grad = False

    print(f"Frozen BERT embeddings + encoder layers 0 to {freeze_until_layer}")


def apply_modality_dropout(batch, p_video=0.2, p_sub=0.45, p_vcpt=0.2):
    B = batch["label"].size(0)
    device = batch["label"].device

    drop_video = torch.rand(B, device=device) < p_video
    if drop_video.any():
        batch["video_feats"][drop_video] = 0.0
        batch["video_mask"][drop_video] = 0

    drop_sub = torch.rand(B, device=device) < p_sub
    if drop_sub.any():
        batch["sub_input_ids"][drop_sub] = 0
        batch["sub_attention_mask"][drop_sub] = 0

    drop_vcpt = torch.rand(B, device=device) < p_vcpt
    if drop_vcpt.any():
        batch["vcpt_obj_ids"][drop_vcpt] = 0
        batch["vcpt_obj_mask"][drop_vcpt] = 0

    return batch


def compute_router_entropy_loss(output, valid, lambda_router_entropy):
    router_weights = output["router_weights"]  # [B, 5, 3]
    router_weights = router_weights[valid]

    router_entropy = -(
        router_weights * torch.log(router_weights + 1e-8)
    ).sum(dim=-1).mean()

    router_loss = -lambda_router_entropy * router_entropy

    return router_loss, router_entropy.detach()


def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total = 0
    correct = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            batch = move_to_device(batch, device)

            labels = batch["label"]
            valid = labels >= 0

            if valid.sum().item() == 0:
                continue

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

            scores = output["scores"]
            scores = scores[valid]
            labels = labels[valid]

            loss = criterion(scores, labels)
            preds = scores.argmax(dim=1)

            total_loss += loss.item() * labels.size(0)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    if total == 0:
        return 0.0, 0.0

    return total_loss / total, correct / total


def save_training_plots(train_log, output_dir):
    if len(train_log) == 0:
        return

    df = pd.DataFrame(train_log)

    csv_path = os.path.join(output_dir, "metrics_log.csv")
    df.to_csv(csv_path, index=False)
    print("Saved metrics CSV →", csv_path)

    plt.figure()
    plt.plot(df["epoch"], df["train_acc"], label="Train Accuracy")
    plt.plot(df["epoch"], df["val_acc"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)

    acc_path = os.path.join(output_dir, "accuracy_curve.png")
    plt.savefig(acc_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved accuracy curve →", acc_path)

    plt.figure()
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
    plt.plot(df["epoch"], df["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    loss_path = os.path.join(output_dir, "loss_curve.png")
    plt.savefig(loss_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved loss curve →", loss_path)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    os.makedirs(args.output_dir, exist_ok=True)

    best_ckpt_path = os.path.join(args.output_dir, "best.pt")
    last_ckpt_path = os.path.join(args.output_dir, "last.pt")
    log_path = os.path.join(args.output_dir, "train_log.json")

    object_vocab = build_object_vocab(
        [
            args.tvqa_vcpt_pkl,
            args.star_vcpt_pkl,
        ],
        min_freq=args.object_min_freq,
        max_objects=args.max_object_vocab,
    )

    print("Object vocab size:", len(object_vocab))

    tvqa_train = UnifiedVQADataset(
        json_path=args.tvqa_train_json,
        siglip_pkl_path=args.tvqa_siglip_pkl,
        vcpt_pkl_path=args.tvqa_vcpt_pkl,
        object_vocab=object_vocab,
        max_frames=args.max_frames,
        max_objects_per_frame=args.max_objects_per_frame,
        siglip_dim=args.siglip_dim,
    )

    star_train = UnifiedVQADataset(
        json_path=args.star_train_json,
        siglip_pkl_path=args.star_siglip_pkl,
        vcpt_pkl_path=args.star_vcpt_pkl,
        object_vocab=object_vocab,
        max_frames=args.max_frames,
        max_objects_per_frame=args.max_objects_per_frame,
        siglip_dim=args.siglip_dim,
    )

    tvqa_val = UnifiedVQADataset(
        json_path=args.tvqa_val_json,
        siglip_pkl_path=args.tvqa_siglip_pkl,
        vcpt_pkl_path=args.tvqa_vcpt_pkl,
        object_vocab=object_vocab,
        max_frames=args.max_frames,
        max_objects_per_frame=args.max_objects_per_frame,
        siglip_dim=args.siglip_dim,
    )

    star_val = UnifiedVQADataset(
        json_path=args.star_val_json,
        siglip_pkl_path=args.star_siglip_pkl,
        vcpt_pkl_path=args.star_vcpt_pkl,
        object_vocab=object_vocab,
        max_frames=args.max_frames,
        max_objects_per_frame=args.max_objects_per_frame,
        siglip_dim=args.siglip_dim,
    )

    train_dataset = ConcatDataset([tvqa_train, star_train])
    val_dataset = ConcatDataset([tvqa_val, star_val])

    print("TVQA train:", len(tvqa_train))
    print("STAR train:", len(star_train))
    print("TVQA val:", len(tvqa_val))
    print("STAR val:", len(star_val))
    print("Total train:", len(train_dataset))
    print("Total val:", len(val_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = RoutedMultimodalVQA(
      num_objects=len(object_vocab),
      bert_name=args.bert_name,
      siglip_dim=args.siglip_dim,
      hidden_dim=768,
      dropout=args.dropout,
      freeze_bert=False,
    ).to(device)

    if args.freeze_lower_bert:
        freeze_lower_bert_layers(
            model,
            freeze_until_layer=args.freeze_until_layer,
        )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    train_log = []

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss = 0.0
        total_ce_loss = 0.0
        total_router_entropy = 0.0
        total = 0
        correct = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

        for batch in pbar:
            batch = move_to_device(batch, device)

            batch = apply_modality_dropout(
                batch,
                p_video=args.p_video_drop,
                p_sub=args.p_sub_drop,
                p_vcpt=args.p_vcpt_drop,
            )

            labels = batch["label"]
            valid = labels >= 0

            if valid.sum().item() == 0:
                continue

            optimizer.zero_grad()

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

            scores = output["scores"]

            scores_valid = scores[valid]
            labels_valid = labels[valid]

            ce_loss = criterion(scores_valid, labels_valid)

            router_loss, router_entropy = compute_router_entropy_loss(
                output=output,
                valid=valid,
                lambda_router_entropy=args.lambda_router_entropy,
            )

            loss = ce_loss + router_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()),
                1.0,
            )
            optimizer.step()

            preds = scores_valid.argmax(dim=1)
            bs = labels_valid.size(0)

            total_loss += loss.item() * bs
            total_ce_loss += ce_loss.item() * bs
            total_router_entropy += router_entropy.item() * bs
            correct += (preds == labels_valid).sum().item()
            total += bs

            pbar.set_postfix(
                {
                    "loss": total_loss / max(total, 1),
                    "ce": total_ce_loss / max(total, 1),
                    "ent": total_router_entropy / max(total, 1),
                    "acc": correct / max(total, 1),
                }
            )

        train_loss = total_loss / max(total, 1)
        train_ce_loss = total_ce_loss / max(total, 1)
        train_router_entropy = total_router_entropy / max(total, 1)
        train_acc = correct / max(total, 1)

        val_loss, val_acc = evaluate(model, val_loader, device)

        print(
            f"\nEpoch {epoch}: "
            f"train_loss={train_loss:.4f}, "
            f"train_ce={train_ce_loss:.4f}, "
            f"router_entropy={train_router_entropy:.4f}, "
            f"train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, "
            f"val_acc={val_acc:.4f}"
        )

        log_item = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_ce_loss": train_ce_loss,
            "train_router_entropy": train_router_entropy,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }

        train_log.append(log_item)

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_acc": val_acc,
            "object_vocab": object_vocab,
            "args": vars(args),
            "train_log": train_log,
        }

        torch.save(ckpt, last_ckpt_path)
        print(f"Saved last checkpoint → {last_ckpt_path}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(ckpt, best_ckpt_path)
            print(f"Saved best checkpoint → {best_ckpt_path}")

        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(train_log, f, indent=2)

        print(f"Saved training log → {log_path}")

    save_training_plots(train_log, args.output_dir)

    print("Training finished.")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print("Best checkpoint:", best_ckpt_path)
    print("Last checkpoint:", last_ckpt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--tvqa_train_json", type=str, default="data/tvqa/tvqa_train_processed.json")
    parser.add_argument("--tvqa_val_json", type=str, default="data/tvqa/tvqa_val_processed.json")
    parser.add_argument("--star_train_json", type=str, default="data/star/star_train_processed.json")
    parser.add_argument("--star_val_json", type=str, default="data/star/star_val_processed.json")

    parser.add_argument("--tvqa_siglip_pkl", type=str, default="data/tvqa/tvqa_siglip_feats.pkl")
    parser.add_argument("--star_siglip_pkl", type=str, default="data/star/star_siglip_feats.pkl")

    parser.add_argument("--tvqa_vcpt_pkl", type=str, default="data/tvqa/vcpt/tvqa_vcpt_yolo.pkl")
    parser.add_argument("--star_vcpt_pkl", type=str, default="data/star/vcpt/star_vcpt_yolo.pkl")

    
    parser.add_argument("--output_dir", type=str,
    default="results/routed_tvqa_star_zeroshot_strongdrop")

    parser.add_argument("--batch_size", type=int, default=2)

    parser.add_argument("--epochs", type=int, default=5)

    parser.add_argument("--lr", type=float, default=1e-6)

    parser.add_argument("--weight_decay", type=float, default=0.01)

    parser.add_argument("--dropout", type=float, default=0.5)

    parser.add_argument("--num_workers", type=int, default=4)

    # modality dropout
    parser.add_argument("--p_video_drop", type=float, default=0.30)
    parser.add_argument("--p_sub_drop", type=float, default=0.60)
    parser.add_argument("--p_vcpt_drop", type=float, default=0.30)

    # router entropy
    parser.add_argument("--lambda_router_entropy", type=float, default=0.05)

    args = parser.parse_args()

    train(args)