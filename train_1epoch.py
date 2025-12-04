"""
EVSSM Task 4: Train the network on GoPro dataset (1 epoch)
This script trains the simplified EVSSM_NoMamba model on the GoPro dataset for 1 epoch.
"""

import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add project to path
sys.path.insert(0, '/home/sjeong7/EVSSM_DL')
from evssm_nomamba import EVSSM_NoMamba


class GoProDataset(Dataset):
    """
    Dataset for GoPro blur-sharp image pairs.
    Structure: data/train/{sequence}/blur/*.png and data/train/{sequence}/sharp/*.png
    """
    def __init__(self, data_root, split='train', max_samples=None, patch_size=256):
        self.data_root = data_root
        self.patch_size = patch_size
        self.pairs = []

        split_dir = os.path.join(data_root, split)
        if not os.path.exists(split_dir):
            raise ValueError(f"Split directory not found: {split_dir}")

        # Iterate through all sequences
        sequences = sorted([d for d in os.listdir(split_dir)
                           if os.path.isdir(os.path.join(split_dir, d))])

        for seq in sequences:
            blur_dir = os.path.join(split_dir, seq, 'blur')
            sharp_dir = os.path.join(split_dir, seq, 'sharp')

            if not os.path.exists(blur_dir) or not os.path.exists(sharp_dir):
                continue

            blur_files = sorted([f for f in os.listdir(blur_dir)
                                if f.endswith(('.png', '.jpg', '.jpeg'))])

            for f in blur_files:
                blur_path = os.path.join(blur_dir, f)
                sharp_path = os.path.join(sharp_dir, f)

                if os.path.exists(sharp_path):
                    self.pairs.append((blur_path, sharp_path))

        if max_samples is not None:
            self.pairs = self.pairs[:max_samples]

        print(f"Found {len(self.pairs)} image pairs in {split} split")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        blur_path, sharp_path = self.pairs[idx]

        # Read images
        blur = cv2.imread(blur_path)
        sharp = cv2.imread(sharp_path)

        if blur is None or sharp is None:
            # Return a random tensor if image loading fails
            return (torch.randn(3, self.patch_size, self.patch_size),
                    torch.randn(3, self.patch_size, self.patch_size))

        # Convert BGR to RGB and normalize to [0, 1]
        blur = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        sharp = cv2.cvtColor(sharp, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # Random crop to patch_size
        h, w, _ = blur.shape
        if h >= self.patch_size and w >= self.patch_size:
            top = np.random.randint(0, h - self.patch_size + 1)
            left = np.random.randint(0, w - self.patch_size + 1)
            blur = blur[top:top+self.patch_size, left:left+self.patch_size]
            sharp = sharp[top:top+self.patch_size, left:left+self.patch_size]
        else:
            # Resize if image is smaller than patch size
            blur = cv2.resize(blur, (self.patch_size, self.patch_size))
            sharp = cv2.resize(sharp, (self.patch_size, self.patch_size))

        # Convert to tensor [C, H, W]
        blur = torch.from_numpy(blur).permute(2, 0, 1)
        sharp = torch.from_numpy(sharp).permute(2, 0, 1)

        return blur, sharp


def train_one_epoch(model, train_loader, optimizer, loss_fn, device):
    """Train for one epoch and return losses."""
    model.train()
    losses = []

    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, (blur, sharp) in enumerate(pbar):
        blur = blur.to(device)
        sharp = sharp.to(device)

        optimizer.zero_grad()
        pred = model(blur)
        loss = loss_fn(pred, sharp)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        # Update progress bar
        avg_loss = np.mean(losses[-20:]) if len(losses) >= 20 else np.mean(losses)
        pbar.set_postfix({'loss': f'{avg_loss:.6f}'})

    return losses


def main():
    print("=" * 60)
    print("EVSSM Task 4: Train on GoPro Dataset (1 Epoch)")
    print("=" * 60)

    # Configuration
    data_root = '/home/sjeong7/EVSSM_DL/data'
    batch_size = 4
    patch_size = 256
    learning_rate = 1e-4
    max_samples = None  # Use all samples for full training; set to e.g., 200 for quick test
    num_workers = 4

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Dataset and DataLoader
    print(f"\nLoading GoPro dataset from: {data_root}")
    train_dataset = GoProDataset(data_root, split='train', max_samples=max_samples, patch_size=patch_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"Total training samples: {len(train_dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Total batches per epoch: {len(train_loader)}")

    # Model
    print(f"\nInitializing EVSSM_NoMamba model...")
    model = EVSSM_NoMamba(dim=48).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # Optimizer and Loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.L1Loss()

    print(f"Optimizer: Adam (lr={learning_rate})")
    print(f"Loss function: L1Loss")

    # Training
    print(f"\n{'=' * 60}")
    print("Starting 1-epoch training...")
    print("=" * 60)

    losses = train_one_epoch(model, train_loader, optimizer, loss_fn, device)

    # Results
    print(f"\n{'=' * 60}")
    print("Training Complete!")
    print("=" * 60)
    print(f"Initial loss: {losses[0]:.6f}")
    print(f"Final loss:   {losses[-1]:.6f}")
    loss_reduction = (1 - losses[-1] / losses[0]) * 100
    print(f"Loss reduction: {loss_reduction:.1f}%")
    print(f"Average loss: {np.mean(losses):.6f}")
    print(f"Min loss: {np.min(losses):.6f}")
    print(f"Max loss: {np.max(losses):.6f}")

    # Save checkpoint
    checkpoint_path = '/home/sjeong7/EVSSM_DL/evssm_nomamba_1epoch.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'final_loss': losses[-1],
        'all_losses': losses,
    }, checkpoint_path)
    print(f"\nCheckpoint saved to: {checkpoint_path}")

    # Plot loss curve (save to file)
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(losses, alpha=0.5, label='Batch Loss')
        # Moving average
        window = min(50, len(losses) // 10)
        if window > 1:
            moving_avg = np.convolve(losses, np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(losses)), moving_avg, 'r-', label=f'Moving Avg (window={window})')
        plt.xlabel('Batch')
        plt.ylabel('L1 Loss')
        plt.title('EVSSM Training Loss (1 Epoch on GoPro)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('/home/sjeong7/EVSSM_DL/training_loss_curve.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Loss curve saved to: /home/sjeong7/EVSSM_DL/training_loss_curve.png")
    except Exception as e:
        print(f"Could not save loss plot: {e}")

    print("\nDone!")


if __name__ == '__main__':
    main()
