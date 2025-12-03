import os
import sys
import torch
import cv2
from torchvision.transforms.functional import to_tensor, to_pil_image

# --- Add project root to path ---
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from models.EVSSM import EVSSM

# ------------ CONFIG ------------
INPUT_DIR = os.path.join(ROOT, "models", "my_samples")
OUTPUT_DIR = os.path.join(ROOT, "outputs")
WEIGHT_PATH = os.path.join(ROOT, "experiments", "pretrained_models", "net_g_GoPro.pth")
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
# --------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_model():
    print(f"Loading EVSSM model on {DEVICE}...")

    model = EVSSM()
    ckpt = torch.load(WEIGHT_PATH, map_location="cpu")

    # Most EVSSM checkpoints store weights under "params"
    if "params" in ckpt:
        state_dict = ckpt["params"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        raise ValueError("No state_dict or params key found in checkpoint.")

    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE)
    model.eval()
    return model


def process_image(model, img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    x = to_tensor(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        y = model(x)

    y = y.squeeze(0).cpu().clamp(0, 1)
    return to_pil_image(y)


def main():
    model = load_model()

    images = sorted(os.listdir(INPUT_DIR))
    print(f"Found {len(images)} images to process.")

    for img_name in images:
        in_path = os.path.join(INPUT_DIR, img_name)
        out_path = os.path.join(OUTPUT_DIR, f"out_{img_name}")

        print("Processing:", img_name)
        out = process_image(model, in_path)
        out.save(out_path)

    print("DONE! Output saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
