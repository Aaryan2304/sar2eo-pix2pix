import os
import torch
import numpy as np
from PIL import Image
from data import tensor_to_pil, get_paired_filename


def save_qualitative_triplets(generator, dataloader, device, output_dir, num_samples=10, epoch=None):
    """Generate SAR → Generated → GT triplets side by side.

    Saves as PNGs: one image per sample, three panels wide.
    """
    os.makedirs(output_dir, exist_ok=True)
    generator.eval()

    with torch.no_grad():
        for batch_idx, (sar, eo) in enumerate(dataloader):
            if batch_idx * sar.size(0) >= num_samples:
                break

            sar = sar.to(device)
            eo = eo.to(device)
            fake_eo = generator(sar)

            for i in range(min(sar.size(0), num_samples - batch_idx * sar.size(0))):
                sar_img = tensor_to_pil(sar[i])
                gen_img = tensor_to_pil(fake_eo[i])
                gt_img = tensor_to_pil(eo[i])

                w, h = sar_img.size
                triplet = Image.new('RGB', (w * 3, h))
                triplet.paste(sar_img.convert('RGB'), (0, 0))
                triplet.paste(gen_img, (w, 0))
                triplet.paste(gt_img, (w * 2, 0))

                suffix = f"_ep{epoch}" if epoch else ""
                triplet.save(os.path.join(output_dir, f'triplet_{batch_idx * sar.size(0) + i:03d}{suffix}.png'))

    generator.train()
    print(f"Qualitative triplets saved to {output_dir}")


def save_single_predictions(generator, dataloader, device, output_dir, num_samples=None):
    """Save individual predictions (for FID computation)."""
    os.makedirs(output_dir, exist_ok=True)
    generator.eval()

    count = 0
    with torch.no_grad():
        for sar, eo in dataloader:
            sar = sar.to(device)
            fake_eo = generator(sar)

            for i in range(sar.size(0)):
                if num_samples and count >= num_samples:
                    break
                img = tensor_to_pil(fake_eo[i])
                img.save(os.path.join(output_dir, f'pred_{count:05d}.png'))
                count += 1

    generator.train()
    print(f"Saved {count} predictions to {output_dir}")
    return count
