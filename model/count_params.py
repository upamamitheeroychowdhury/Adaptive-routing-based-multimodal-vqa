import torch

from model.routed_multimodal_vqa import RoutedMultimodalVQA

# ======================================================
# ALL CHECKPOINTS
# ======================================================

CHECKPOINTS = {
    "Baseline Routed Fusion":
        "results/routed_tvqa_star_temporal_vcpt/best.pt",

    "Modality Dropout + Routing":
        "results/routed_tvqa_star_moddrop_router/best.pt",

    "Strong Dropout":
        "results/routed_tvqa_star_zeroshot_strongdrop/best.pt",

    "Regularized":
        "results/routed_tvqa_star_zeroshot_regularized_run2/best.pt",
}

# ======================================================
# PARAMETER COUNTER
# ======================================================

def count_model_params(model):

    total_params = 0
    trainable_params = 0
    frozen_params = 0

    for param in model.parameters():

        num_params = param.numel()

        total_params += num_params

        if param.requires_grad:
            trainable_params += num_params
        else:
            frozen_params += num_params

    return total_params, trainable_params, frozen_params


# ======================================================
# LOOP OVER CHECKPOINTS
# ======================================================

for model_name, ckpt_path in CHECKPOINTS.items():

    print("\n" + "=" * 80)
    print(f"MODEL: {model_name}")
    print("=" * 80)

    ckpt = torch.load(ckpt_path, map_location="cpu")

    object_vocab = ckpt["object_vocab"]

    model = RoutedMultimodalVQA(
        num_objects=len(object_vocab),
        bert_name="bert-base-uncased",
        siglip_dim=768,
        hidden_dim=768,
        dropout=0.35,
        freeze_bert=False,
    )

    model.load_state_dict(ckpt["model"])

    total_params, trainable_params, frozen_params = count_model_params(model)

    print(f"Checkpoint: {ckpt_path}")
    print(f"Total Parameters      : {total_params:,}")
    print(f"Trainable Parameters  : {trainable_params:,}")
    print(f"Frozen Parameters     : {frozen_params:,}")

print("\nDone.")