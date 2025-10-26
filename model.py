"""
PyTorch training script for an AlexNet-style model on the Imagenette dataset.

This script includes:
- Model definition (AlexNet)
- Data loading and transformation
- Training loop
- Validation loop
- Checkpoint saving and loading
"""

import os
import time
import torch
import torch.nn as nn
import argparse
import torch.optim as optim
import tqdm # For progess bar
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.nn import functional as F
from typing import Tuple

# -----------Config for alexnet + transformer - decoder-block --------------

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
batch_size = 64 # how many independent sequences will we process in parallel?
num_classes = 10
num_epochs = 4
data_root = "data"
checkpoint_path = "alexnet_imagenette.pth"
num_workers = os.cpu_count() // 2 if os.cpu_count() else 4
pin_memory = False if device.type == 'mps' else True
weight_decay = 0.0005
learning_rate = 1e-4
n_embd = 256
n_head = 16
n_layer = 16
dropout = 0.5

# ------------ Config for collect activation of model ---------------------- 

DATASET_PATH = 'data' # IMPORTANT: Set this path
OUTPUT_DIR = 'activation_dataset_block6_resid' # Directory to save activations
TARGET_BLOCK_INDEX = 6 # 7th block is index 6 (Residual Stream Output)
NUM_WORKERS = os.cpu_count() // 2 if os.cpu_count() else 4
PIN_MEMORY = False if device.type == 'mps' else True
IMG_SIZE = 224 # Image size your model expects
# -----------------Class names for decoding----------------
CLASS_NAMES = [
    "tench", "English springer", "cassette player", "chain saw", "church", 
    "French horn", "garbage truck", "gas pump", "golf ball", "parachute"
]


# --- Helper: Ensure Output Directory Exists ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Saving activations to: {os.path.abspath(OUTPUT_DIR)}")
print(f"Using device: {device}")

#---------------Model---------------------

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPT(nn.Module):

    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd * 36, num_classes)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T, C = idx.shape

        x = self.blocks(idx) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        x = torch.flatten(x,1)
        output = self.lm_head(x) # (B,T,vocab_size)

        return output

# --- 2. Model Definition ---

class AlexNet(nn.Module):
    """
    AlexNet-style architecture adapted for smaller inputs and a specific 
    number of classes.
    """
    def __init__(self, num_classes: int = 1000) -> None:
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(96),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.transformer = GPT()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x) #Output: [B, 256, 6, 6]
        x = self.avgpool(x) #Output: [B, 256, 6, 6]
        # Reshape for transformer: treat the 6x6 grid as a sequence of 36 'patches'
        # [B, C, H, W] -> [B, C, H*W] -> [B, H*W, C]
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(0,2,1) # Output: [B, 36, 256]
        # Now x has shape [Batch, SequenceLenght = 36, EmbeddingDim = 256]
        x = self.transformer(x) # Pass through transformer blocks and final head
        return x


# --- 3. Helper Functions ---

def get_data_loaders(
    root: str, 
    batch_size: int, 
    num_workers: int, 
    pin_memory: bool
) -> Tuple[DataLoader, DataLoader]:
    """Creates and returns the training and validation DataLoaders."""
    print("Loading Imagenette dataset...")
    
    # Standard normalization for ImageNet-trained models
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.Imagenette(root=root,
                                        split='train',
                                        transform=transform,
                                        download=True)

    val_dataset = datasets.Imagenette(root=root,
                                      split='val',
                                      transform=transform,
                                      download=True)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=pin_memory)
    
    print("Datasets loaded successfully.")
    return train_loader, val_loader


#save the model
def save_checkpoint(
    model: nn.Module, 
    optimizer: optim.Optimizer, 
    epoch: int, 
    loss: float, 
    history: dict,
    path: str
):
    """Saves the model checkpoint."""
    print(f"✅ Saving checkpoint at epoch {epoch+1} to {path}")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'history': history, 
    }, path)

#load the checkpoint
def load_checkpoint(
    model: nn.Module, 
    optimizer: optim.Optimizer, 
    path: str,
    device: torch.device
) -> Tuple[int, float, dict]:
    """Loads the model checkpoint."""
    print(f"🔄 Loading checkpoint from {path}")
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    loss = checkpoint['loss']
    
    history = checkpoint.get('history',{
        'train_loss':[],'train_acc':[], 'val_loss':[],'val_acc':[]
    })

    print(f"Resumed from epoch {start_epoch} (last saved loss: {loss:.4f})")
    return start_epoch, loss, history

# Actual training function
def train_one_epoch(
    model: nn.Module, 
    loader: DataLoader, 
    optimizer: optim.Optimizer, 
    criterion: nn.Module, 
    device: torch.device
) -> Tuple[float, float]:
    """Runs one full training epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100
    return epoch_loss, epoch_acc

#validation to check model performance
def validate(
    model: nn.Module, 
    loader: DataLoader, 
    criterion: nn.Module, 
    device: torch.device
) -> Tuple[float, float]:
    """Runs one full validation epoch."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100
    return epoch_loss, epoch_acc

# ---Hook Setup----

activation_storage = {'activation': None}
hook_handle = None

def get_activation_hook(name):
    """Hook function to capture output and store it."""
    def hook(module, input, output): # Correct arguments: module, input, output
        #Block output is the residual stream value for that block
        if isinstance(output, tuple):
            activation_storage[name] = output[0].detach().cpu()
        else:
            activation_storage[name] = output.detach().cpu()

        return hook
         

# --- 4. Main Execution ---
def main():
    """Main training and validation script."""
    parser = argparse.ArgumentParser(description='Train AlexNet on Imagenette')
    parser.add_argument(
        '--train',
        type=bool,
        default=False,
        help='Train the model'
    )
    parser.add_argument(
        '-e', '--epochs', 
        type=int, 
        default=2, 
        help='Total number of epochs to train for (default: 2)'
    )
    parser.add_argument(
        '--store_activation',
        type=bool,
        default=False,
        help='Store the activation from the layer you want!'
    )
    parser.add_argument(
        '--inference',
        type=bool,
        default=False,
        help='inference the model'
    )
    args = parser.parse_args()
    num_epochs = args.epochs  # This is our new "total epochs"
    train = args.train
    store_activation = args.store_activation
    inference = args.inference
    print("hello",train)

    # For Training the Model.
    if train == True and store_activation == False :
        print(f"--- AlexNet Training on Imagenette ---")
        print(f"Using device: {device}")
        print(f"Using {num_workers} workers and Pin Memory: {pin_memory}")

        # --- 1. Load Data ---
        train_loader, val_loader = get_data_loaders(
            root=data_root,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

        # --- 2. Initialize Model, Loss, and Optimizer ---
        model = AlexNet(num_classes=num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # Print model summary
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model initialized with {total_params/1e6:.2f}M trainable parameters.")

        # --- 3. Load Checkpoint (if exists) ---
        start_epoch = 0
        best_val_loss=float('inf')
        history = {
            'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []
        }
        if os.path.exists(checkpoint_path):
            try:
                start_epoch, loss , history = load_checkpoint(
                    model, optimizer, checkpoint_path, device
                )
            except Exception as e:
                print(f"Warning: Could not load checkpoint. Starting fresh. Error: {e}")
        else:
            print("No checkpoint found, starting fresh.")

        if start_epoch >= num_epochs:
                print(f"\nModel has already been trained for {start_epoch} epochs. Target is {num_epochs}.")
                print("To train for more epochs, increase the --epochs argument.")

        # --- 4. The Training Loop ---
        else:
            print(f"\n--- Starting Training from Epoch {start_epoch+1} ---")
            total_start_time = time.time()

            for epoch in range(start_epoch, num_epochs):  # <-- USE num_epochs
                epoch_start_time = time.time()
                
                # Train
                train_loss, train_acc = train_one_epoch(
                    model, train_loader, optimizer, criterion, device
                )
                
                # Validate
                val_loss, val_acc = validate(
                    model, val_loader, criterion, device
                )
                
                # Save metrics for plotting
                history['train_loss'].append(train_loss)
                history['train_acc'].append(train_acc)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                epoch_duration = time.time() - epoch_start_time
                
                # Log results
                print(f"Epoch {epoch+1:02}/{num_epochs} | "  # <-- USE num_epochs
                    f"Time: {epoch_duration:.2f}s | "
                    f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                    f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
                
                # Save checkpoint only if validation loss has improved
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(
                        model, optimizer, epoch, val_loss, history, checkpoint_path
                    )
                else:
                    print(f"Validation loss did not improve from {best_val_loss:.4f}.")

            total_end_time = time.time()
            print("\n--- Training Finished ---")
            print(f"Total training time: {(total_end_time - total_start_time) / 60:.2f} minutes")
            print(f"Best validation loss: {best_val_loss:.4f}")
            print(f"Final model and checkpoint saved to {checkpoint_path}")


    if inference == True and train == False:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"Saving activations to: {os.path.abspath(OUTPUT_DIR)}")
        print(f"Using device: {device}")

        print("Loading model and weights...")
        model = AlexNet(num_classes=num_classes).to(device)

        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
                print("Loaded weights from 'model_state_dict' key.")
            else:
                model.load_state_dict(checkpoint)
                print("Loaded weights directly from checkpoint file.")
        except FileNotFoundError:
            print(f"ERROR: Checkpoint file not found at {checkpoint_path}. Cannot load weights.")
            exit()
        # ... (rest of your try/except blocks for loading)
        
        model.eval()
        print("Model loaded successfully.")
        
        # --- Data Loading and Preprocessing ---
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        print("Loading dataset...")
        try:
            image_dataset = datasets.ImageFolder(DATASET_PATH, transform=preprocess)
            dataloader = DataLoader(image_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
            print(f"Dataset loaded with {len(image_dataset)} images.")
        except FileNotFoundError:
            print(f"ERROR: Dataset not found at {DATASET_PATH}. Please set the correct path.")
            exit()
        #-- put the inference code here---

        # # --- Hook Setup ---
        # global activation_storage, hook_handle # Declare as global if hook function is outside main
        # activation_storage = {'activation': None}
        # hook_handle = None
        if store_activation:
            try:
                if TARGET_BLOCK_INDEX < 0 or TARGET_BLOCK_INDEX >= n_layer:
                    raise IndexError(f"TARGET_BLOCK_INDEX ({TARGET_BLOCK_INDEX}) out of range for {n_layer} layers.")
                target_module = model.transformer.blocks[TARGET_BLOCK_INDEX]
                hook_handle = target_module.register_forward_hook(get_activation_hook('activation'))
                print(f"Hook registered on output of transformer block {TARGET_BLOCK_INDEX}.")
            except Exception as e:
                print(f"An unexpected error occurred during hook registration: {e}")
                exit()
            # --- Activation Collection Loop ---
            print("Starting activation collection...")
            saved_count = 0
            file_indices = {}
            original_filepaths = [item[0] for item in image_dataset.samples]

            with torch.no_grad():
                for batch_idx, (inputs, _) in enumerate(tqdm.tqdm(dataloader, desc="Processing Batches")):
                    inputs = inputs.to(device)
                    try:
                        _ = model(inputs)
                    except Exception as e:
                        print(f"\nError during forward pass on batch {batch_idx}: {e}")
                        continue

                    if activation_storage['activation'] is None:
                        print(f"Warning: Hook did not capture any activation in batch {batch_idx}.")
                        continue

                    batch_activations = activation_storage['activation']
                    current_batch_size = batch_activations.shape[0]
                    
                    for i in range(current_batch_size):
                        dataset_idx = batch_idx * dataloader.batch_size + i
                        if dataset_idx >= len(original_filepaths):
                            continue

                        single_activation = batch_activations[i]
                        original_filename = os.path.basename(original_filepaths[dataset_idx])
                        filename_base, _ = os.path.splitext(original_filename)

                        if filename_base not in file_indices:
                            file_indices[filename_base] = 0
                        else:
                            file_indices[filename_base] += 1
                        instance_idx = file_indices[filename_base]

                        save_filename = f"activation_block{TARGET_BLOCK_INDEX}_{filename_base}_{instance_idx:02d}.pt"
                        save_path = os.path.join(OUTPUT_DIR, save_filename)
                        
                        try:
                            torch.save(single_activation, save_path)
                            saved_count += 1
                        except Exception as e:
                            print(f"\nError saving activation for index {dataset_idx} ({save_filename}): {e}")

                    activation_storage['activation'] = None

            # --- Cleanup ---
            if hook_handle:
                hook_handle.remove()
                print("Hook removed.")

            print(f"\nActivation collection complete.")
            print(f"Successfully saved {saved_count} activation tensors to '{OUTPUT_DIR}'.")

            # --- Optional: Verify saved file ---
            if saved_count > 0:
                try:
                    saved_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.pt')]
                    if saved_files:
                        first_file = os.path.join(OUTPUT_DIR, saved_files[0])
                        loaded_tensor = torch.load(first_file)
                        print(f"Verified saving: Loaded '{saved_files[0]}' with shape {loaded_tensor.shape}")
                        if loaded_tensor.shape != torch.Size([36, n_embd]):
                            print(f"Warning: Loaded tensor shape {loaded_tensor.shape} does not match expected [36, {n_embd}]")
                    else:
                        print("Could not verify saving: No .pt files found in output directory.")
                except Exception as e:
                    print(f"Error during verification load: {e}")

    if store_activation == False and train == False:   
        print("Provide task you want to do : --train=True or --store_activation=True or --inference=True" )

    # # --- 5. Plot and Save Curves ---
    # # This will now plot the *full* history, even when resuming
    # if not history['train_loss']:
    #      print("\nNo training was performed, skipping plot generation.")
    # else:
    #     plot_save_path = 'training_curves.png'
    #     plot_curves(history, plot_save_path)
    #     print(f"Training curves saved to {plot_save_path}")
    #
if __name__ == '__main__':
    main()
