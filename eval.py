import os
import argparse
import json

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import UNetGenerator
from data import SEN12Dataset, load_splits, get_paired_filename
from utils.metrics import compute_lpips, compute_ssim, compute_psnr, compute_fid
from utils.visualization import save_qualitative_triplets, save_single_predictions


def evaluate(generator, dataloader, device, pred_save_dir=None):
    """Run generator on a dataloader, compute LPIPS/SSIM/PSNR.

    If pred_save_dir is set, also saves individual predictions for FID.
    """
    generator.eval()

    all_lpips, all_ssim, all_psnr = [], [], []

    if pred_save_dir:
        os.makedirs(pred_save_dir, exist_ok=True)

    with torch.no_grad():
        for batch_idx, (sar, eo) in enumerate(tqdm(dataloader, desc="Evaluating", leave=False)):
            sar = sar.to(device)
            eo = eo.to(device)
            fake_eo = generator(sar)

            all_lpips.append(compute_lpips(fake_eo, eo))
            all_ssim.append(compute_ssim(fake_eo, eo))
            all_psnr.append(compute_psnr(fake_eo, eo))

            if pred_save_dir:
                for i in range(sar.size(0)):
                    from data import tensor_to_pil
                    img = tensor_to_pil(fake_eo[i])
                    img.save(os.path.join(pred_save_dir, f'pred_{batch_idx * sar.size(0) + i:05d}.png'))

    results = {
        'lpips': sum(all_lpips) / len(all_lpips),
        'ssim': sum(all_ssim) / len(all_ssim),
        'psnr': sum(all_psnr) / len(all_psnr)
    }

    return results


def save_ground_truth(dataloader, save_dir):
    """Save ground truth EO images for FID comparison."""
    os.makedirs(save_dir, exist_ok=True)
    from data import tensor_to_pil

    count = 0
    for batch_idx, (sar, eo) in enumerate(dataloader):
        for i in range(eo.size(0)):
            img = tensor_to_pil(eo[i])
            img.save(os.path.join(save_dir, f'gt_{count:05d}.png'))
            count += 1
    print(f"Saved {count} ground truth images to {save_dir}")


def run_eval(config_path, weights_path, experiment_name, device='cuda'):
    """Full evaluation pipeline — metrics, FID, qualitative."""
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    exp_dir = os.path.join('./outputs', experiment_name)
    os.makedirs(exp_dir, exist_ok=True)

    # Load generator
    G = UNetGenerator(
        in_channels=config['data']['sar_channels'],
        out_channels=config['data']['eo_channels'],
        base_filters=config['model']['generator']['base_filters'],
        num_layers=config['model']['generator']['num_layers'],
        use_dropout=config['model']['generator']['use_dropout'],
        dropout_rate=config['model']['generator']['dropout_rate']
    ).to(device)

    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    G.load_state_dict(state_dict)
    G.eval()
    print(f"Loaded weights from {weights_path}")

    # Load data
    root = config['data']['root_dir']
    classes = config['data']['classes']
    split_path = os.path.join(root, 'splits.json')
    splits = load_splits(split_path)

    val_ds = SEN12Dataset(root, splits['val'], classes)
    test_ds = SEN12Dataset(root, splits['test'], classes)

    batch_size = config['training']['batch_size']
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    # Evaluate
    print("\n--- Validation ---")
    val_pred_dir = os.path.join(exp_dir, 'val_predictions')
    val_results = evaluate(G, val_loader, device, pred_save_dir=val_pred_dir)

    print("\n--- Test ---")
    test_pred_dir = os.path.join(exp_dir, 'test_predictions')
    test_results = evaluate(G, test_loader, device, pred_save_dir=test_pred_dir)

    # FID
    print("\nComputing FID...")
    val_gt_dir = os.path.join(exp_dir, 'val_gt')
    test_gt_dir = os.path.join(exp_dir, 'test_gt')
    save_ground_truth(val_loader, val_gt_dir)
    save_ground_truth(test_loader, test_gt_dir)

    val_fid = compute_fid(val_pred_dir, val_gt_dir, batch_size=50, device=device)
    test_fid = compute_fid(test_pred_dir, test_gt_dir, batch_size=50, device=device)

    val_results['fid'] = float(val_fid)
    test_results['fid'] = float(test_fid)

    # Float-ify everything so JSON doesn't choke on numpy types
    for d in (val_results, test_results):
        for k in d:
            d[k] = float(d[k])

    # Qualitative
    print("\nGenerating qualitative triplets...")
    save_qualitative_triplets(G, test_loader, device,
                              os.path.join(exp_dir, 'qualitative'), num_samples=10)

    # Print results
    print(f"\n{'='*60}")
    print(f"Results for {experiment_name}")
    print(f"{'='*60}")
    print(f"{'Metric':<12} {'Val':>12} {'Test':>12}")
    print(f"{'-'*36}")
    for metric in ['lpips', 'fid', 'ssim', 'psnr']:
        v = val_results.get(metric, 0)
        t = test_results.get(metric, 0)
        fmt = '{:.4f}' if metric in ['lpips', 'ssim'] else '{:.2f}'
        print(f"{metric:<12} {fmt.format(v):>12} {fmt.format(t):>12}")

    # Save
    all_results = {'val': val_results, 'test': test_results}
    with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {os.path.join(exp_dir, 'results.json')}")
    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to generator checkpoint')
    parser.add_argument('--experiment', type=str, required=True,
                        help='Experiment name')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    run_eval(args.config, args.weights, args.experiment, device)


if __name__ == '__main__':
    main()
