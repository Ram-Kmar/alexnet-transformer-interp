# Vision Model Interpretability via Sparse Autoencoders

> An end-to-end pipeline to identify and analyze conceptual features in vision models using a Sparse Autoencoder (SAE) on the transformer's residual stream.

## Overview

This repository provides a framework for training and applying **Sparse Autoencoders (SAEs)** to the internal activations of vision transformer models. The primary goal is to decompose a model's high-dimensional representations—specifically from the residual stream—into more sparse, monosemantic, and interpretable conceptual features.

The methodology is informed by current interpretability research, with methods drawing inspiration from recent advancements in feature analysis (e.g., techniques discussed in the Gemma Scope paper).

---

## ⚠️ Project Status: Early Development

**This project is in its early stages.** The codebase is functional but subject to change as the implementation evolves.

Users and potential contributors are encouraged to **read the source code** to fully understand the current implementation, especially if you intend to build upon or adapt this work.

---

## Usage

The main script (`model.py`) is controlled via command-line arguments to select the desired operational mode.

### 1. Train a Custom Model

To train the vision model on a specified dataset (e.g., Imagenette), use the `--train` flag.
```bash
python model.py --train=True

### 2. Epochs 
python model.py -e 10

###3. Inference
python model.py --inference=True

###4 collect activations
python model.py --inference=True --store_activation=True
