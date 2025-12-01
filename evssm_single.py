"""
evssm_single.py
Simplified single-file PyTorch training script for EVSSM assignment.

This file replicates the basic behaviour of the EVSSM repository:
- dataset loading
- model definition
- training loop
- validation with PSNR
- optional inference

But without Basicsr, YAML, or any heavy framework.

To use the REAL EVSSM model, replace the placeholder network
in class EVSSMNet with the architecture from models/ in the EVSSM repo.
"""

import os
from glob import glob
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms



# 1. EVSSM NETWORK


class EVSSMNet(nn.Module):
    """
    Replace this block with the real EVSSM architecture.
    For now, we use a tiny CNN so the pipeline is fully runnable.
    """
    def __init__(self):
        super().__init__()

        # TODO: paste real EVSSM architecture here
        # e.g. from models/evssm_arch.py in original repo

        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=1),
        )

    def forward(self, x):
        return self.net(x)



# 2. DATASET CLASS

class PairedDeblurDataset(Dataset):
    """
    Expects a folder structure like:

    dataset/
        blur/
            img001.png
            img002.png
        sharp/
            img001.png
            img002.png

    blur[i] corresponds to sharp[i].

    """

    def __init__(self, root, transform=None):
        self.blur_paths = sorted(glob(os.path.join(root, "blur", "*")))
        self.sharp_paths = sorted(glob(os.path.join(root, "sharp", "*")))

        assert len(self.blur_paths) == len(self.sharp_paths), \
            "Blur & Sharp folder must have same number of images"

        self.transform = transform

    def __len__(self):
        return len(self.blur_paths)

    def __getitem__(self, idx):
        blur = Image.open(self.blur_paths[idx]).convert("RGB")
        sharp = Image.open(self.sharp_paths[idx]).convert("RGB")

        if self.transform:
            blur = self.transform(blur)
            sharp = self.transform(sharp)

        return {
            "blur": blur,
            "sharp": sharp,
            "blur_path": self.blur_paths[idx],
            "sharp_path": self.sharp_paths[idx],
        }



# 3. METRICS (PSNR)


def psnr(pred, gt, max_val=1.0):
    mse = torch.mean((pred - gt) ** 2)
    if mse == 0:
        return torch.tensor(float("inf"))
    return 20 * torch.log10(max_val / torch.sqrt(mse))

# 4. TRAINING FUNCTION


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    criterion = nn.L1Loss()
    running_loss = 0

    for i, batch in enumerate(dataloader, start=1):
        blur = batch["blur"].to(device)
        sharp = batch["sharp"].to(device)

        pred = model(blur)
        loss = criterion(pred, sharp)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 10 == 0:
            print(f"[TRAIN] Step {i}, Loss = {loss.item():.4f}")

    return running_loss / len(dataloader)



# 5. VALIDATION FUNCTION


@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()
    criterion = nn.L1Loss()
    total_loss = 0
    total_psnr = 0
    count = 0

    for batch in dataloader:
        blur = batch["blur"].to(device)
        sharp = batch["sharp"].to(device)

        pred = model(blur)
        loss = criterion(pred, sharp)
        total_loss += loss.item()

        pred_clamped = torch.clamp(pred, 0, 1)
        sharp_clamped = torch.clamp(sharp, 0, 1)

        for i in range(pred.size(0)):
            total_psnr += psnr(pred_clamped[i], sharp_clamped[i]).item()

        count += pred.size(0)

    return total_loss / len(dataloader), total_psnr / count



# 6. INFERENCE ON A FOLDER OF BLURRED IMAGES

@torch.no_grad()
def run_inference(model, input_dir, output_dir, device, img_size=256):
    os.makedirs(output_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    model.eval()
    img_paths = sorted(glob(os.path.join(input_dir, "*")))

    print(f"[INFER] Found {len(img_paths)} images")

    for path in img_paths:
        img = Image.open(path).convert("RGB")
        t = transform(img).unsqueeze(0).to(device)

        pred = model(t)
        pred = torch.clamp(pred.squeeze(), 0, 1).cpu()

        out = transforms.ToPILImage()(pred)
        save_path = os.path.join(output_dir, os.path.basename(path))
        out.save(save_path)

        print(f"[INFER] Saved {save_path}")




# 7. MAIN SCRIPT


def main():

    # --------  EDIT THESE PATHS --------
    train_dir = "./data/GoPro/train"   # folder with blur/sharp
    val_dir   = "./data/GoPro/val"     # folder with blur/sharp
    # -------------------------------------------

    img_size = 256
    batch_size = 4
    lr = 1e-4
    epochs = 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    train_ds = PairedDeblurDataset(train_dir, transform)
    val_ds   = PairedDeblurDataset(val_dir, transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = EVSSMNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs+1):
        print(f"\n===== EPOCH {epoch}/{epochs} =====")
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_psnr = validate(model, val_loader, device)

        print(f"[RESULT] Train Loss = {train_loss:.4f}")
        print(f"[RESULT] Val   Loss = {val_loss:.4f}")
        print(f"[RESULT] Val PSNR   = {val_psnr:.2f} dB")

    # Save checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = "checkpoints/evssm_single.pth"
    torch.save(model.state_dict(), ckpt_path)
    print(f"\nSaved model to {ckpt_path}")


if __name__ == "__main__":
    main()
