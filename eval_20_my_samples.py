"""
EVSSM Task 4 Extension: Evaluate trained model on 20 my_samples images
Compare outputs before training (random init) vs after training (1 epoch on GoPro)
"""

import os
import cv2
import torch
import numpy as np
from evssm_nomamba import EVSSM_NoMamba

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

base_dir = "/home/sjeong7/EVSSM_DL"
samples_dir = os.path.join(base_dir, "my_samples")
out_dir_trained = os.path.join(base_dir, "my_samples_outputs_trained")
out_dir_untrained = os.path.join(base_dir, "my_samples_outputs_untrained")

os.makedirs(out_dir_trained, exist_ok=True)
os.makedirs(out_dir_untrained, exist_ok=True)

# ============================================================
# 1. Load TRAINED model (after 1 epoch on GoPro)
# ============================================================
model_trained = EVSSM_NoMamba(dim=48).to(device)
ckpt_path = os.path.join(base_dir, "evssm_nomamba_1epoch.pth")
ckpt = torch.load(ckpt_path, map_location=device)

# Handle checkpoint format (may have 'model_state_dict' key or be raw state_dict)
if 'model_state_dict' in ckpt:
    model_trained.load_state_dict(ckpt['model_state_dict'])
else:
    model_trained.load_state_dict(ckpt)

model_trained.eval()
print(f"✓ Loaded TRAINED model from: {ckpt_path}")

# ============================================================
# 2. Load UNTRAINED model (random initialization for comparison)
# ============================================================
model_untrained = EVSSM_NoMamba(dim=48).to(device)
model_untrained.eval()
print("✓ Loaded UNTRAINED model (random init)")

# ============================================================
# 3. Get 20 sample images
# ============================================================
files = sorted([f for f in os.listdir(samples_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))])[:20]

print(f"\nFound {len(files)} sample images in {samples_dir}")
print("=" * 60)

# ============================================================
# 4. Run inference on both models
# ============================================================
def run_inference(model, img_rgb, device):
    """Run model inference on an RGB image."""
    h, w, _ = img_rgb.shape

    # Resize to 256x256 for model input
    inp = cv2.resize(img_rgb, (256, 256))
    inp_t = torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(inp_t)
        pred = torch.clamp(pred, 0.0, 1.0)
        pred_np = pred[0].permute(1, 2, 0).cpu().numpy()

    # Resize back to original size
    pred_resized = cv2.resize(pred_np, (w, h))
    return pred_resized


for idx, name in enumerate(files):
    in_path = os.path.join(samples_dir, name)

    # Read image
    img = cv2.imread(in_path)
    if img is None:
        print(f"❌ Could not read: {name}")
        continue

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # Run inference with TRAINED model
    pred_trained = run_inference(model_trained, img_rgb, device)
    pred_trained_bgr = cv2.cvtColor((pred_trained * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)

    # Run inference with UNTRAINED model
    pred_untrained = run_inference(model_untrained, img_rgb, device)
    pred_untrained_bgr = cv2.cvtColor((pred_untrained * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)

    # Save outputs
    base_name = os.path.splitext(name)[0]
    ext = os.path.splitext(name)[1]

    out_path_trained = os.path.join(out_dir_trained, f"{base_name}_trained{ext}")
    out_path_untrained = os.path.join(out_dir_untrained, f"{base_name}_untrained{ext}")

    cv2.imwrite(out_path_trained, pred_trained_bgr)
    cv2.imwrite(out_path_untrained, pred_untrained_bgr)

    print(f"[{idx+1:2d}/20] {name}")
    print(f"        → Trained:   {out_path_trained}")
    print(f"        → Untrained: {out_path_untrained}")

print("=" * 60)
print(f"\n✓ Done! Results saved to:")
print(f"  - Trained outputs:   {out_dir_trained}")
print(f"  - Untrained outputs: {out_dir_untrained}")
print(f"\nYou can now visually compare:")
print(f"  1. Original blur images in: {samples_dir}")
print(f"  2. Untrained model outputs (random init)")
print(f"  3. Trained model outputs (1 epoch on GoPro)")
