Adaptive Routing-based Multimodal Video Question Answering for Zero-shot Cross-Dataset Generalization

Official implementation of our lightweight multimodal Video Question Answering (VQA) framework for zero-shot cross-dataset generalization across TVQA, STAR, and NExT-QA.

GitHub Repository:
Adaptive-routing-based-multimodal-vqa

Overview

This project proposes a lightweight adaptive-routing multimodal VQA architecture designed to improve zero-shot transfer across different video QA datasets.

Unlike large Vision-Language Models (VLMs), the proposed framework uses:

BERT-based textual reasoning
SigLIP visual embeddings
VCPT/object-level semantic representations
Adaptive routed multimodal fusion
Entropy-regularized modality balancing
Subtitle dropout regularization

The model is trained on:

TVQA
STAR

and evaluated in a zero-shot manner on:

NExT-QA

without any fine-tuning on the target dataset.

Key Contributions
Adaptive Routed Fusion

The framework dynamically balances contributions from:

subtitles
video features
object-level VCPT features

through a learnable routing mechanism.

Zero-shot Cross-Dataset Generalization

The model is trained on TVQA + STAR and directly evaluated on NExT-QA without target-domain adaptation.

Subtitle Bias Reduction

Strong subtitle dropout regularization is introduced to reduce over-reliance on textual shortcuts and improve visual grounding.

Lightweight Multimodal Architecture

Compared to large VLM-based systems, the proposed framework remains computationally lightweight while still improving cross-dataset transfer performance.

Architecture

The overall pipeline is:

Video Frames
     ↓
SigLIP Encoder
     ↓
Temporal Video Features
     ↓
------------------------------------

Subtitles
     ↓
BERT Encoder
     ↓
Subtitle Features
     ↓
------------------------------------

VCPT / Object Labels
     ↓
BERT Encoder
     ↓
Object Features
     ↓
------------------------------------

Question + Candidate Answers
     ↓
BERT Encoder
     ↓
Question-Answer Representation
     ↓

Adaptive Routing Fusion
(Modality-aware weighted fusion)

     ↓

Answer Prediction Head
     ↓

Multiple-choice Answer Selection
Datasets
TVQA

Source:
TVQA Dataset

Used for:

supervised multimodal VQA training
STAR

Source:
STAR Dataset

Used for:

additional multimodal training
improving reasoning diversity
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
Training Variants
Model Variant	Train Accuracy	Validation Accuracy	Zero-shot NExT-QA
Baseline Routed Fusion	45.23%	44.73%	23.77%
ModDrop + Router Entropy	50.89%	44.73%	24.04%
Strong Subtitle Dropout	50.20%	43.79%	24.10%
Regularized Run2	47.20%	43.04%	23.88%
Main Findings
Subtitle-heavy models show weaker cross-dataset transfer.
Modality balancing improves generalization.
Strong subtitle regularization improves zero-shot transfer performance.
Adaptive routing reduces modality dominance.
Lightweight multimodal fusion can remain competitive without large-scale VLMs.
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
│
├── results/
│
└── checkpoints/
Training

Example training command:

python train/train_tvqa_star_zeroshot_strongdrop.py
Testing

Example zero-shot evaluation:

python test/test_nextqa_routed.py
Output Files

Testing automatically saves:

nextqa_predictions.json
nextqa_predictions.csv
confusion_matrix.csv
confusion_matrix.png
classification_report.json

inside the corresponding results directory.

Parameter Count

Approximate total parameters:

134.9M parameters
Future Work

Possible future extensions include:

temporal grounding supervision
reasoning alignment loss
timestamp-aware fusion
VLM integration
domain-invariant contrastive learning
adaptive temporal routing
Citation

If you use this repository, please cite:

@misc{adaptive_routed_vqa_2026,
  title={Adaptive Routing-based Multimodal Video Question Answering for Zero-shot Cross-Dataset Generalization},
  author={Upama Roy Chowdhury},
  year={2026}
}
Contact

Upama Roy Chowdhury
Mechanical Engineering Lecturer & AI Researcher

GitHub:
GitHub Profile
