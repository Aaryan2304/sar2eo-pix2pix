import os
import argparse
from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image
import numpy as np

from models import UNetGenerator


def load_generator(weights_path, device='cuda'):
    """Load trained generator weights for inference."""
    G = UNetGenerator(in_channels=1, out_channels=3, base_filters=64, num_layers=8,
                      use_dropout=True, dropout_rate=0.5)

    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    G.load_state_dict(state_dict)
    G.to(device)
    G.eval()
    print(f"Generator loaded from {weights_path}")
    return G


def preprocess_sar(image_path, image_size=256):
    """Load a single SAR patch and normalise to [-1, 1].

    Expects: 256x256 single-channel PNG, dB-scaled, [0, 255].
    Returns: tensor [1, 1, 256, 256] in [-1, 1].
    """
    img = Image.open(image_path).convert('L')
    img = img.resize((image_size, image_size), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    tensor = (tensor / 255.0 - 0.5) / 0.5
    return tensor


def postprocess_eo(tensor):
    """Generator output [-1, 1] → PIL Image [0, 255] RGB."""
    tensor = (tensor + 1.0) / 2.0
    tensor = tensor.clamp(0, 1)
    arr = (tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


@torch.no_grad()
def run_inference(input_dir, output_dir, weights_path, device='cuda'):
    """Run inference on all SAR PNGs in input_dir, save EO outputs to output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    G = load_generator(weights_path, device)

    input_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith('.png')])
    print(f"Found {len(input_files)} input images")

    for fname in input_files:
        input_path = os.path.join(input_dir, fname)
        output_path = os.path.join(output_dir, fname)

        sar_tensor = preprocess_sar(input_path).to(device)
        eo_tensor = G(sar_tensor)
        eo_img = postprocess_eo(eo_tensor)
        eo_img.save(output_path)

    print(f"Inference complete. {len(input_files)} images saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='SAR-to-EO Inference')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory of single-channel SAR PNG patches')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save generated EO RGB PNG images')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to generator checkpoint (.pth)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device: cuda or cpu')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    run_inference(args.input_dir, args.output_dir, args.weights, device)


if __name__ == '__main__':
    main()
