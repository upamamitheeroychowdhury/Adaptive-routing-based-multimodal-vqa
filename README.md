# Adaptive Routing-based Multimodal Video Question Answering for Zero-shot Cross-Dataset Generalization

Official implementation of a lightweight multimodal Video Question Answering (VQA) framework for zero-shot cross-dataset transfer across TVQA, STAR, and NExT-QA.

---

## Repository

GitHub Repository:  
https://github.com/upamamitheeroychowdhury/Adaptive-routing-based-multimodal-vqa

---

# Overview

This project proposes a lightweight adaptive-routing multimodal VQA framework designed to improve zero-shot cross-dataset generalization.

The framework combines:

- BERT-based textual reasoning
- SigLIP visual representations
- VCPT/object-level semantic representations
- Adaptive routed multimodal fusion
- Entropy-regularized modality balancing
- Subtitle dropout regularization

The model is trained on:

- TVQA
- STAR

and evaluated directly on:

- NExT-QA

without any target-domain fine-tuning.

---

# Key Contributions

## Adaptive Routed Fusion

A learnable routing mechanism dynamically balances:

- subtitle features
- visual features
- object-level VCPT features

during multimodal fusion.

---

## Zero-shot Cross-Dataset Generalization

The framework is trained on TVQA + STAR and directly evaluated on NExT-QA without adaptation.

---

## Subtitle Bias Reduction

Strong subtitle dropout regularization reduces over-reliance on textual shortcuts and improves visual grounding.

---

## Lightweight Multimodal Framework

Unlike large Vision-Language Models (VLMs), the proposed framework remains computationally lightweight while improving cross-dataset transfer performance.

---

# Architecture

## Overall Pipeline

```text
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
