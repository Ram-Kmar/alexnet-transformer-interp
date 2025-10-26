import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import os
import glob
import tqdm
from typing import Tuple
import argparse

# --- 1. Configuration ---
# --- Paths (UPDATE THESE) ---
ACTIVATION_DIR = 'activation_dataset_block6_resid' # Directory where you saved activations
MODEL_SAVE_PATH = 'sae_block6_jump_relu.pth' # Path to save the trained SAE

# --- Model Hyperparameters ---
# d_model matches the embedding dim of your transformer activations
D_MODEL = 256
# d_sae is the "dictionary size".
D_SAE = D_MODEL * 8  # e.g., 2048


# --- NEW: JumpReLU & L0 Hyperparameters ---
# L0_COEFFICIENT is lambda in the paper. It balances recon vs. sparsity.
# This value is HIGHLY sensitive and you will need to sweep it.
# Start with a value that makes l0_loss comparable to mse_loss.
L0_COEFFICIENT = 1e-3 
# EPSILON is the "kernel density estimator bandwidth" from the paper.
# It controls the smoothness of the surrogate loss for training theta.
EPSILON = 0.001 

# --- Training Hyperparameters ---
LEARNING_RATE = 1e-4 # JumpReLU often requires a smaller LR than standard ReLU
BATCH_SIZE = 128
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# print(f"Using device: {DEVICE}")
# print(f"Loading activations from: {os.path.abspath(ACTIVATION_DIR)}")
# print(f"SAE dimensions: d_model={D_MODEL}, d_sae={D_SAE}")

@torch.no_grad()
def get_dataset_variance(dataloader: DataLoader, device: torch.device) -> torch.Tensor:
    """
    Calculates the variance of the entire dataset (or a large sample)
    to normalize the MSE loss (for FVU)
    """
    print("Calculating dataset variance(for FVU)...")
    all_acts = []

    # Iterate over a large sample (e.Next, 100 batches) to get a good estimate
    # This is much faster than using the whole dataset.
    num_batches_for_variance = 1000

    for i, batch_data in enumerate(dataloader):
        if i >= num_batches_for_variance:
            break
        all_acts.append(batch_data.to(device))

    if not all_acts:
        print("Error: Dataloader is empty, cannot calculate variance")
        return torch.tensor(1.0) # Return 1.0 to avoid division by zero

    # Concatenate all batches into one big tensor
    all_acts_tensor  = torch.cat(all_acts)

    #Reshape from [B, seq_len, d_model] to [N, d_model] where N = B * Seq_Len
    all_acts_tensor = all_acts_tensor.view(-1, all_acts_tensor.shape[-1])

    # Calculate the variance over all tokens and dimensions
    variance = torch.var(all_acts_tensor)

    print(f"Dataset variance calculated:{variance.item():.4e}")
    return variance


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
        print(f"Found {len(self.file_paths)} activation files.")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Load a single activation tensor
        return torch.load(self.file_paths[idx])

# --- 3. JumpReLU Sparse Autoencoder (MODIFIED) ---
class JumpReLUSparseAutoencoder(nn.Module):
    def __init__(self, d_model, d_sae, epsilon=EPSILON):
        super().__init__()
        
        self.W_enc = nn.Parameter(torch.empty(d_model, d_sae))
        nn.init.kaiming_uniform_(self.W_enc, a=0.01)
        
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_model))
        nn.init.kaiming_uniform_(self.W_dec, a=0.01)
        
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        
        # --- NEW: JumpReLU Parameters ---
        # This is the learnable threshold vector theta from the paper
        self.theta = nn.Parameter(torch.empty(d_sae))
        # Initialize theta to a small positive value (as per paper)
        nn.init.constant_(self.theta, 0.001)
        
        # Store epsilon as a non-learnable buffer
        self.register_buffer("epsilon", torch.tensor(epsilon))
        # --- End New ---
        
        self.normalize_decoder_weights()

    @torch.no_grad()
    def normalize_decoder_weights(self):
        # Normalize columns of W_dec to have L2 norm of 1
        self.W_dec.data = F.normalize(self.W_dec.data, p=2, dim=0)

    def encode(self, input_acts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        input_acts shape: [..., d_model] (e.g., [B, 36, 256])
        
        Returns:
            sae_acts (torch.Tensor): The *gated* activations (f(x))
            pre_acts (torch.Tensor): The *pre-gated* activations (z)
        """
        # z = W_enc * x + b_enc
        pre_acts = input_acts @ self.W_enc + self.b_enc
        
        # --- NEW: JumpReLU Activation ---
        # This is the "hard gate" H(z - theta)
        # We use .float() to create a 0/1 mask
        gate = (pre_acts > self.theta).float()
        
        # f(x) = z * H(z - theta)
        sae_acts = pre_acts * gate
        # --- End New ---
        
        # We return both:
        # - sae_acts: for the reconstruction loss (trains W_enc, W_dec, etc.)
        # - pre_acts: for the surrogate L0 loss (trains theta)
        return sae_acts, pre_acts

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        """
        acts shape: [..., d_sae] (e.g., [B, 36, 2048])
        """
        return acts @ self.W_dec + self.b_dec

    def forward(self, input_acts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            recon (torch.Tensor): Reconstructed activations [..., d_model]
            sae_acts (torch.Tensor): Internal SAE activations (gated) [..., d_sae]
            pre_acts (torch.Tensor): Internal SAE activations (pre-gate) [..., d_sae]
        """
        sae_acts, pre_acts = self.encode(input_acts)
        recon = self.decode(sae_acts)
        return recon, sae_acts, pre_acts

# --- 4. Main Training Function (MODIFIED FOR FVU) ---
def train_sae():
    # --- Setup Data ---
    try:
        dataset = ActivationDataset(ACTIVATION_DIR)
    except FileNotFoundError:
        print("Stopping due to missing data.")
        return
        
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=os.cpu_count() // 2,
        pin_memory=True
    )

    # --- Setup Model ---
    model = JumpReLUSparseAutoencoder(D_MODEL, D_SAE).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # --- NEW: Calculate Dataset Variance for FVU ---
    # We do this once at the beginning.
    dataset_variance = get_dataset_variance(dataloader, DEVICE)
    # --- END NEW ---

    print("Starting JumpReLU SAE training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        total_mse_loss = 0.0
        total_l0_loss = 0.0
        total_l0_surr = 0.0

        progress_bar = tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch_data in progress_bar:
            batch_data = batch_data.to(DEVICE)

            # --- Forward Pass ---
            recon_acts, sae_acts, pre_acts = model(batch_data)

            # --- Calculate Loss ---
            
            # 1. Reconstruction Loss (MSE)
            mse_loss = F.mse_loss(recon_acts, batch_data)

            # 2a. "True" L0 Loss (for logging)
            true_l0_per_vector = torch.sum(sae_acts > 1e-9, dim=-1).float()
            true_l0 = torch.mean(true_l0_per_vector)

            # 2b. Surrogate L0 Loss (for training theta)
            x = (pre_acts - model.theta) / model.epsilon
            surrogate_gate = torch.sigmoid(x)
            surrogate_l0_per_vector = torch.sum(surrogate_gate, dim=-1)
            surrogate_l0_loss = torch.mean(surrogate_l0_per_vector)

            # 3. Total Loss
            loss = mse_loss + L0_COEFFICIENT * surrogate_l0_loss

            # --- Backward Pass ---
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # --- Renormalize Decoder Weights ---
            model.normalize_decoder_weights()
            
            # --- NEW: Calculate FVU for this batch (for logging) ---
            batch_fvu = mse_loss.item() / dataset_variance.item()
            # --- END NEW ---

            # --- Logging ---
            total_loss += loss.item()
            total_mse_loss += mse_loss.item() # Accumulate MSE
            total_l0_loss += true_l0.item()
            total_l0_surr += surrogate_l0_loss.item()
            
            progress_bar.set_postfix(
                loss=f"{loss.item():.4e}",
                mse=f"{mse_loss.item():.4e}",
                FVU=f"{batch_fvu:.4f}",  # --- NEW: Log batch FVU
                L0=f"{true_l0.item():.2f}",
                L0_surr=f"{surrogate_l0_loss.item():.2f}"
            )

        # --- End of Epoch ---
        avg_loss = total_loss / len(dataloader)
        avg_mse = total_mse_loss / len(dataloader) # This is the average MSE
        avg_l0 = total_l0_loss / len(dataloader)
        avg_l0_surr = total_l0_surr / len(dataloader)
        
        # --- NEW: Calculate Average FVU for the Epoch ---
        avg_fvu = avg_mse / dataset_variance.item()
        # --- END NEW ---
        
        # --- MODIFIED: Added Avg FVU to the print statement ---
        print(f"  Avg Loss: {avg_loss:.4e} | Avg MSE: {avg_mse:.4e} | Avg FVU: {avg_fvu:.4f} | Avg L0: {avg_l0:.2f}")

    # --- Save the trained model ---
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nTraining complete. Model saved to {MODEL_SAVE_PATH}")

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
    recon_acts, sae_acts , pre_acts= model(batch_data)

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

def main():
    parser = argparse.ArgumentParser(description='Train AlexNet on Imagenette')
    parser.add_argument(
        '--train',
        type=bool,
        default=False,
        help='Train the model'
    )
    # parser.add_argument(
    #     '-e', '--epochs', 
    #     type=int, 
    #     default=2, 
    #     help='Total number of epochs to train for (default: 2)'
    # )
    parser.add_argument(
        '--inference',
        type=bool,
        default=False,
        help='inference the model!'
    )
    args = parser.parse_args()
    # num_epochs = args.epochs  # This is our new "total epochs"
    train = args.train
    inference = args.inference

    if train == True and inference == False:
        train_sae()
    elif train == False and inference == True:
        run_inference()
    elif train == False and inference==False:
        print("Provide task you want to do : --train=True or --inference=True")

if __name__ == '__main__':
    main()