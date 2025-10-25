import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import os
import glob
import tqdm
from typing import Tuple

# --- 1. Configuration (Set for Inference) ---
# --- Paths ---
ACTIVATION_DIR = 'activation_dataset_block6_resid' # Directory to get activations for analysis
MODEL_SAVE_PATH = 'sae_block6_jump_relu.pth' # Path to *load* the trained SAE from

# --- Model Hyperparameters (Must match the trained model) ---
D_MODEL = 256
D_SAE = D_MODEL * 8  # e.g., 2048
EPSILON = 0.001 

# --- Inference Hyperparameters ---
BATCH_SIZE = 128 # Can be any size
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

print(f"Using device: {DEVICE}")
print(f"Loading trained model from: {MODEL_SAVE_PATH}")


# --- 2. Custom Dataset for Activations (Unchanged) ---
class ActivationDataset(Dataset):
    """
    Loads .pt activation files from a directory.
    Each file is expected to be a tensor of shape [seq_len, d_model],
    e.g., [36, 256].
    """
    def __init__(self, data_dir):
        self.file_paths = glob.glob(os.path.join(data_dir, "*.pt"))
        if not self.file_paths:
            print(f"Error: No .pt files found in {data_dir}")
            raise FileNotFoundError
        print(f"Found {len(self.file_paths)} activation files for inference.")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Load a single activation tensor
        return torch.load(self.file_paths[idx])

# --- 3. JumpReLU Sparse Autoencoder (Cleaned for Inference) ---
class JumpReLUSparseAutoencoder(nn.Module):
    def __init__(self, d_model, d_sae, epsilon=EPSILON):
        super().__init__()
        
        # --- Parameters (structure must match saved file) ---
        self.W_enc = nn.Parameter(torch.empty(d_model, d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        self.theta = nn.Parameter(torch.empty(d_sae))
        
        self.register_buffer("epsilon", torch.tensor(epsilon))

    def encode(self, input_acts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the sparse activations and the pre-gate activations.
        """
        pre_acts = input_acts @ self.W_enc + self.b_enc
        gate = (pre_acts > self.theta).float()
        sae_acts = pre_acts * gate
        return sae_acts, pre_acts # Return pre_acts in case you want to analyze it

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        """
        Decodes the sparse activations back to the original space.
        """
        return acts @ self.W_dec + self.b_dec

    def forward(self, input_acts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inference-focused forward pass.
        Returns:
            recon (torch.Tensor): Reconstructed activations [..., d_model]
            sae_acts (torch.Tensor): Internal SAE activations (gated) [..., d_sae]
        """
        sae_acts, _ = self.encode(input_acts) # We don't need pre_acts for inference
        recon = self.decode(sae_acts)
        return recon, sae_acts

# --- 4. NEW: Inference Function ---
@torch.no_grad() # Disable gradient calculation
def run_inference():
    # --- Setup Data ---
    try:
        dataset = ActivationDataset(ACTIVATION_DIR)
    except FileNotFoundError:
        print("Stopping due to missing data.")
        return
        
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False, # No need to shuffle for inference
        num_workers=os.cpu_count() // 2,
        pin_memory=True
    )

    # --- Load Model ---
    print(f"Loading model structure: d_model={D_MODEL}, d_sae={D_SAE}")
    model = JumpReLUSparseAutoencoder(D_MODEL, D_SAE).to(DEVICE)
    
    try:
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_SAVE_PATH}")
        print("Please train the model first by running the training script.")
        return
    except Exception as e:
        print(f"Error loading model state dict: {e}")
        print("Ensure D_MODEL and D_SAE in this script match the trained model.")
        return
        
    model.eval() # Set model to evaluation mode
    
    print("\n--- Starting Inference ---")

    # --- Get one batch to analyze ---
    try:
        batch_data = next(iter(dataloader))
        batch_data = batch_data.to(DEVICE)
    except StopIteration:
        print("Error: Dataloader is empty. Cannot run inference.")
        return

    # --- Run data through the SAE ---
    # recon_acts: The model's reconstruction
    # sae_acts: The sparse feature activations (the "concepts")
    recon_acts, sae_acts = model(batch_data)

    # --- Analyze the results ---
    
    # 1. Batch-level statistics
    batch_mse = F.mse_loss(recon_acts, batch_data).item()
    true_l0_per_vector = torch.sum(sae_acts > 1e-9, dim=-1).float()
    avg_l0 = torch.mean(true_l0_per_vector).item()
    
    print(f"\n--- Batch Statistics ---")
    print(f"Input shape:   {batch_data.shape}")
    print(f"SAE act shape: {sae_acts.shape}")
    print(f"Recon shape:   {recon_acts.shape}")
    print(f"Avg MSE:       {batch_mse:.4e}")
    print(f"Avg L0 (Sparsity): {avg_l0:.2f} active features (out of {D_SAE})")

    # 2. Token-level analysis (This is the "neuron-to-concept" part)
    # Let's inspect the very first token in the batch
    # Shape: [seq_len, d_model] -> [36, 256]
    first_act_in_batch = batch_data[0] 
    # Shape: [seq_len, d_sae] -> [36, 2048]
    first_sae_acts_in_batch = sae_acts[0] 
    
    # Let's look at the first token (e.g., at sequence position 0)
    token_to_analyze = first_sae_acts_in_batch[0] # Shape: [2048]
    
    # Find which features are active (non-zero)
    l0_this_token = torch.sum(token_to_analyze > 1e-9).item()
    
    # Get the 10 *strongest* active features
    # This tells you which "concepts" were most present in this token
    top_values, top_indices = torch.topk(token_to_analyze, k=10)
    
    print(f"\n--- Analysis of a Single Token (Batch 0, Token 0) ---")
    print(f"Total active features (L0) for this token: {l0_this_token}")
    print("Top 10 strongest feature activations:")
    print("Feature Index | Activation Value")
    print("-" * 30)
    for value, index in zip(top_values, top_indices):
        if value.item() > 1e-9: # Only print if it's truly active
            print(f"  {index.item():<11} | {value.item():.4f}")

if __name__ == '__main__':
    run_inference()