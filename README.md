
Adaptive Routing-based Multimodal Video Question Answering for Zero-shot Cross-Dataset Generalization

Official implementation of a lightweight adaptive-routing multimodal VQA framework for zero-shot cross-dataset transfer across TVQA, STAR, and NExT-QA.

Repository:
Adaptive-routing-based-multimodal-vqa

Overview

This project proposes a lightweight multimodal Video Question Answering (VQA) framework that improves zero-shot cross-dataset generalization using:

Adaptive routed multimodal fusion
Subtitle dropout regularization
Entropy-regularized modality balancing
SigLIP visual representations
BERT-based textual reasoning
VCPT/object-level semantic representations

The model is trained on:

TVQA
STAR

and evaluated directly on:

NExT-QA

without target-domain fine-tuning.

Key Contributions
Adaptive Routed Fusion

A learnable routing module dynamically balances:

subtitle features
visual features
object-level VCPT features

during multimodal fusion.

Zero-shot Cross-Dataset Transfer

The framework is trained on TVQA + STAR and evaluated directly on NExT-QA without adaptation.

Subtitle Bias Reduction

Strong subtitle dropout regularization reduces over-reliance on textual shortcuts and improves visual grounding.

Lightweight Multimodal Framework

Unlike large Vision-Language Models (VLMs), the proposed framework remains computationally efficient while still improving cross-dataset transfer performance.

Architecture
Overall Pipeline
Video Frames
    │
    ▼
SigLIP Encoder
    │
    ▼
Temporal Video Features


Subtitles
    │
    ▼
BERT Encoder
    │
    ▼
Subtitle Features


VCPT / Object Labels
    │
    ▼
BERT Encoder
    │
    ▼
Object Features


Question + Candidate Answers
    │
    ▼
BERT Encoder
    │
    ▼
Question-Answer Representation


Adaptive Routing Fusion
(Modality-aware weighted fusion)
    │
    ▼
Answer Prediction Head
    │
    ▼
Multiple-choice Answer Selection
Datasets
TVQA

Source:
TVQA Dataset

Used for:

multimodal VQA training
STAR

Source:
STAR Dataset

Used for:

additional multimodal training
reasoning diversity
NExT-QA

Source:
NExT-QA Dataset

Used for:

zero-shot evaluation
Features Used
Visual Features
SigLIP (google/siglip-base-patch16-224)
32 sampled frames per video
Object Features
YOLO-based VCPT representations
Textual Features
subtitles / ASR
question-answer candidate pairs

Encoded using:

bert-base-uncased
Experimental Results
Model Variant	Train Accuracy	Validation Accuracy	Zero-shot NExT-QA
Baseline Routed Fusion	45.23%	44.73%	23.77%
ModDrop + Router Entropy	50.89%	44.73%	24.04%
Strong Subtitle Dropout	50.20%	43.79%	24.10%
Regularized Run2	47.20%	43.04%	23.88%
Main Findings
Subtitle-heavy models show weaker cross-dataset transfer.
Adaptive modality balancing improves zero-shot generalization.
Strong subtitle regularization improves transfer performance.
Lightweight multimodal fusion remains competitive without large VLMs.
Project Structure
URC_PROJECT/
│
├── data/
│   ├── tvqa/
│   ├── star/
│   └── nextqa/
│
├── model/
│   ├── routed_multimodal_vqa.py
│   ├── routed_multimodal_vqa_subclamp_routerlog.py
│   └── bert_baseline.py
│
├── train/
│   ├── train_tvqa_star.py
│   ├── train_tvqa_star_realignment.py
│   ├── train_tvqa_star_zeroshot_strongdrop.py
│   └── train_routed_tvqa_star_zeroshot_regularized.py
│
├── test/
│   └── test_nextqa_routed.py
│
├── preprocess/
├── results/
└── checkpoints/
Training

Example:

python train/train_tvqa_star_zeroshot_strongdrop.py
Testing

Example:

python test/test_nextqa_routed.py
Evaluation Outputs

The testing pipeline automatically saves:

nextqa_predictions.json
nextqa_predictions.csv
confusion_matrix.csv
confusion_matrix.png
classification_report.json

inside the corresponding results directory.

Model Size

Approximate parameter count:

134.9M parameters
Future Work

Possible future extensions:

temporal grounding supervision
reasoning alignment loss
timestamp-aware fusion
domain-invariant multimodal alignment
adaptive temporal routing
lightweight VLM integration
Citation
@misc{adaptive_routed_vqa_2026,
  title={Adaptive Routing-based Multimodal Video Question Answering for Zero-shot Cross-Dataset Generalization},
  author={Upama Roy Chowdhury},
  year={2026}
}
Contact

Upama Roy Chowdhury

GitHub:
GitHub Profile
