# Vision Model Interpretability via Sparse Autoencoders

> An end-to-end pipeline to identify and analyze conceptual features in vision models using a Sparse Autoencoder (SAE) on the transformer's residual stream.

## Overview

This repository provides a framework for training and applying **Sparse Autoencoders (SAEs)** to the internal activations of vision transformer models. The project uses a hybrid model composed of an **AlexNet** backbone for feature extraction and a **Transformer decoder** for processing these features.

The primary goal is to decompose the model's high-dimensional representations—specifically from the residual stream of the transformer blocks—into more sparse, monosemantic, and interpretable "conceptual" features. By analyzing which features activate on which parts of an image, we can gain a better understanding of what the model has learned.

The methodology is inspired by recent advancements in interpretability research, particularly the use of SAEs to uncover meaningful features within large language models, and applies them to the vision domain.

---

## ⚠️ Project Status: Early Development

**This project is in its early stages.** The codebase is functional but subject to change as the implementation evolves.

Users and potential contributors are encouraged to **read the source code** to fully understand the current implementation, especially if you intend to build upon or adapt this work.

---

## Pipeline

The core of this project is a multi-stage pipeline:

1.  **Train the Vision Model:** An AlexNet-style CNN followed by a Transformer decoder is trained on an image classification task (e.g., Imagenette).
2.  **Collect Activations:** The trained model is run in inference mode on a dataset of images. We use a forward hook to capture the internal activations from a specific layer (e.g., the residual stream of a transformer block). These activations are saved to disk.
3.  **Train the Sparse Autoencoder (SAE):** A sparse autoencoder is trained on the collected activations. The SAE learns to reconstruct the activations from a compressed, sparse representation. This project uses a **JumpReLU SAE**, which encourages sparsity through a learnable threshold.
4.  **Analyze Features:** The trained SAE can then be used to analyze new images. By looking at which of the SAE's internal features activate most strongly for a given input, we can identify the "concepts" the model is using to make its decisions.

## Usage

### 1. Setup

First, clone the repository and install the required dependencies. It is recommended to use a virtual environment.

```bash
git clone https://github.com/your-username/alexnet-transformer-interp.git
cd alexnet-transformer-interp
pip install -r requirements.txt # You may need to create this file
```

The training script will automatically download the Imagenette dataset to the `data/` directory.

### 2. Train the Vision Model

The `model.py` script is used to train the main vision model.

```bash
python model.py --train=True --epochs=10
```

-   `--train=True`: Specifies that you want to run the training process.
-   `--epochs=10`: The total number of epochs to train for.

This will train the model on the Imagenette dataset and save the best-performing checkpoint to `alexnet_imagenette.pth`.

### 3. Collect Activations

After training, use the same script to run inference and collect activations. You will need to place your images for inference in a directory (e.g., `images/`).

First, ensure the configuration at the top of `model.py` is set correctly:

-   `DATASET_PATH`: Path to your folder of images for activation collection.
-   `OUTPUT_DIR`: Directory where the activation tensors will be saved.
-   `TARGET_BLOCK_INDEX`: The index of the transformer block from which to capture activations.

Then, run the script:

```bash
python model.py --inference=True --store_activation=True
```

This will process the images in `DATASET_PATH`, capture the activations from the specified transformer block, and save each activation as a separate `.pt` file in `OUTPUT_DIR`.

### 4. Train the Sparse Autoencoder

With the activations collected, you can now train the SAE using `sae.py`.

First, check the configuration at the top of `sae.py`:

-   `ACTIVATION_DIR`: Should match the `OUTPUT_DIR` from the previous step.
-   `MODEL_SAVE_PATH`: The path where the trained SAE model will be saved.

Then, run the training script:

```bash
python sae.py --train=True
```

This will train the `JumpReLUSparseAutoencoder` on the activation dataset and save the final model to `MODEL_SAVE_PATH`. The training process includes logging the Mean Squared Error (MSE), Fraction of Variance Unexplained (FVU), and the L0 norm (sparsity) of the SAE's features.

### 5. Analyze SAE Features

To analyze the learned features, you can run the `sae.py` script in inference mode.

```bash
python sae.py --inference=True
```

This will load the trained SAE, run a batch of activations through it, and print an analysis for a single token. The output will show the top 10 strongest feature activations for that token, giving you an insight into which "concepts" the SAE has learned.

## Code Structure

-   `model.py`: Contains the main vision model (AlexNet + Transformer), data loaders for Imagenette, and the logic for training, inference, and activation collection.
-   `sae.py`: Contains the `JumpReLUSparseAutoencoder` implementation, a custom `Dataset` for loading activations, and the training and inference logic for the SAE.
-   `activation_dataset_block6_resid/`: The default directory for storing collected activations.
-   `data/`: The default directory for the Imagenette dataset.

## Model Architecture

### Vision Model

The main model in `model.py` is a combination of a convolutional feature extractor and a transformer.

1.  **AlexNet Features:** The `features` part of the `AlexNet` class is a standard AlexNet-style convolutional network that processes the input image and extracts a grid of feature vectors.
2.  **Transformer:** The output of the CNN is reshaped into a sequence of embeddings (from `[B, C, H, W]` to `[B, H*W, C]`) and fed into a `GPT`-style transformer decoder. This allows the model to learn relationships between different spatial parts of the image.
3.  **Classification Head:** A final linear layer maps the transformer's output to the number of classes for classification.

### Sparse Autoencoder

The `JumpReLUSparseAutoencoder` in `sae.py` is designed to learn a sparse basis for the activations from the vision model.

-   **Encoder/Decoder:** It has a standard autoencoder structure with an encoder that maps the input activations to a higher-dimensional space (`d_sae`) and a decoder that reconstructs the original activations from this space.
-   **JumpReLU Activation:** Instead of a standard ReLU, it uses a "JumpReLU" activation function. This is a gated ReLU where the gate is a hard threshold `H(z - theta)`, and `theta` is a *learnable* parameter. This encourages sparsity by forcing small activations to zero.
-   **Surrogate L0 Loss:** To train the `theta` parameter, the model uses a surrogate L0 loss function based on a sigmoid approximation, which is differentiable. The total loss is a combination of the reconstruction MSE and this surrogate L0 loss.
-   **Decoder Weight Normalization:** The weights of the decoder are normalized to have an L2 norm of 1, which is a common practice in SAE training.