# Quantum-Enhanced Text Sentiment Analysis

> **A PyTorch and PennyLane implementation of the text-modality pipeline from the paper *"QeMMA: Quantum-Enhanced Multi-Modal Sentiment Analysis"***

## Overview

This repository presents a research-oriented implementation of the **text branch** of the QeMMA architecture, integrating classical deep learning with variational quantum computing for sentiment classification.

The implementation reproduces the text-processing pipeline described in the original paper by combining a Bidirectional GRU, attention mechanism, and a parameterized quantum circuit for hybrid quantum-classical learning.

**Note:** This repository focuses exclusively on the **text modality**. The visual and multimodal fusion components described in the original paper are not included.

---

## Architecture

```text
Input Text
     │
     ▼
Tokenization & Vocabulary Construction
     │
     ▼
Embedding Layer
     │
     ▼
Bidirectional GRU
     │
     ▼
Attention Layer
     │
     ▼
Linear Projection
     │
     ▼
Parameterized Quantum Circuit
(Angle Embedding + Strongly Entangling Layers)
     │
     ▼
Quantum Measurement
     │
     ▼
Linear Classifier
     │
     ▼
Sentiment Prediction
```

---

## Implemented Components

* Embedding-based text representation
* Bidirectional GRU encoder
* Attention mechanism for contextual feature aggregation
* Hybrid quantum layer implemented using PennyLane
* Variational Quantum Circuit (VQC) with trainable parameters
* End-to-end PyTorch training pipeline
* Data preprocessing and vocabulary construction
* Sequence padding and batching
* Model checkpointing with training resumption
* Validation after every epoch
* Macro F1-score evaluation

---

## Technical Stack

* Python
* PyTorch
* PennyLane
* NumPy
* Pandas
* Scikit-learn

---

## Dataset

Experiments are performed on a Twitter sentiment analysis dataset.

Sentiment labels are mapped as:

| Original Label | Mapped Class |
| -------------- | ------------ |
| -1             | Negative (0) |
| 0              | Neutral (1)  |
| 1              | Positive (2) |

---

## Training Pipeline

The training workflow includes:

* Data preprocessing
* Vocabulary construction
* Integer encoding
* Sequence padding
* Train/validation split
* Mini-batch training
* Validation after each epoch
* Macro F1-score computation
* Automatic checkpoint saving
* Resume training from saved checkpoints

---

## Quantum Layer

The hybrid quantum module consists of:

* Angle Embedding
* Strongly Entangling Layers
* Expectation-value measurement over four qubits
* Integration with PyTorch through PennyLane's `TorchLayer`

The quantum circuit acts as a learnable feature transformation layer before the final classifier.

---

## Repository Structure

```text
.
├── model.py
├── train.py
├── README.md
├── checkpoints/
└── dataset/
```

---

## Current Status

* Text-modality architecture implemented
* End-to-end training pipeline
* Validation pipeline
* Checkpointing and training resumption
* Macro F1-score evaluation
* Hyperparameter tuning in progress
* Experimental result comparison with the original paper

---

## Disclaimer

This repository is an **independent research implementation** of the **text modality** presented in the paper **"QeMMA: Quantum-Enhanced Multi-Modal Sentiment Analysis."** It is intended for educational and research purposes and is **not** an official implementation from the original authors.

---

## Future Work

* Reproduce the complete multimodal architecture
* Incorporate the image processing branch
* Implement multimodal feature fusion
* Benchmark against the experimental results reported in the original paper
* Explore alternative variational quantum circuit designs and quantum embedding strategies
