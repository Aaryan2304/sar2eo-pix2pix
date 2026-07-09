import os, time, argparse, json, csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models import UNetGenerator, PatchGANDiscriminator
from data import get_dataloaders, load_config, tensor_to_pil, SEN12Dataset, create_splits, load_splits
from utils.metrics import compute_lpips, compute_ssim, compute_psnr
from utils.visualization import save_qualitative_triplets


def train(config_path, experiment_name, resume=False):
    config = load_config(config_path)
    exp_config = config['experiment']
    train_config = config['training']
    model_config = config['model']
    loss_config = config['loss']

    device = torch.device(exp_config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    exp_dir = os.path.join('./outputs', experiment_name)
    ckpt_dir = os.path.join(exp_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    train_loader, val_loader, test_loader = get_dataloaders(config, seed=exp_config['seed'])

    G = UNetGenerator(
        in_channels=config['data']['sar_channels'],
        out_channels=config['data']['eo_channels'],
        base_filters=model_config['generator']['base_filters'],
        num_layers=model_config['generator']['num_layers'],
        use_dropout=model_config['generator']['use_dropout'],
        dropout_rate=model_config['generator']['dropout_rate']
    ).to(device)

    use_gan = 'gan' in experiment_name

    if use_gan:
        D = PatchGANDiscriminator(
            in_channels=config['data']['sar_channels'] + config['data']['eo_channels'],
            base_filters=model_config['discriminator']['base_filters'],
            num_layers=model_config['discriminator']['num_layers']
        ).to(device)

    lr_g, lr_d = train_config['lr_generator'], train_config['lr_discriminator']
    beta1, beta2 = train_config['beta1'], train_config['beta2']

    opt_g = Adam(G.parameters(), lr=lr_g, betas=(beta1, beta2))
    if use_gan:
        opt_d = Adam(D.parameters(), lr=lr_d, betas=(beta1, beta2))

    total_epochs = train_config['epochs']
    decay_start = train_config['lr_schedule']['decay_start_epoch']

    def lr_lambda(epoch):
        if epoch < decay_start:
            return 1.0
        progress = (epoch - decay_start) / (total_epochs - decay_start)
        return max(0.0, 1.0 - progress)

    scheduler_g = LambdaLR(opt_g, lr_lambda)
    if use_gan:
        scheduler_d = LambdaLR(opt_d, lr_lambda)

    l1_loss = nn.L1Loss()
    if use_gan:
        bce_loss = nn.BCEWithLogitsLoss()

    history = {
        'train': {'g_loss': [], 'd_loss': [], 'l1_loss': []},
        'val': {'l1_loss': [], 'ssim': [], 'psnr': []}
    }

    best_val_l1 = float('inf')
    best_epoch = 0
    lambda_l1 = loss_config['l1_weight']
    log_interval = train_config['log_interval']
    val_interval = train_config['val_interval']
    start_epoch = 1
    global_step = 0

    # meta.json persists the batch_size used during training so resume
    # can correctly reconstruct the epoch from global_step. Only written once.
    meta_path = os.path.join(exp_dir, 'meta.json')
    if not os.path.exists(meta_path):
        json.dump({
            'batch_size': train_config['batch_size'],
            'total_epochs': total_epochs,
            'use_gan': use_gan,
        }, open(meta_path, 'w'))

    if resume:
        last_path = os.path.join(ckpt_dir, 'gen_last.pth')
        if os.path.exists(last_path):
            try:
                G.load_state_dict(torch.load(last_path, map_location=device, weights_only=True))
                print(f"Resumed generator from {last_path}")
            except Exception:
                print("Warning: gen_last.pth corrupted, trying best checkpoint...")
                best_path = os.path.join(ckpt_dir, 'gen_best.pth')
                if os.path.exists(best_path):
                    G.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
                    print(f"Resumed generator from {best_path}")

            if use_gan:
                disc_last = os.path.join(ckpt_dir, 'disc_last.pth')
                if os.path.exists(disc_last):
                    try:
                        D.load_state_dict(torch.load(disc_last, map_location=device, weights_only=True))
                    except Exception:
                        print("Warning: disc_last.pth corrupted")

            opt_last = os.path.join(ckpt_dir, 'opt_last.pth')
            if os.path.exists(opt_last):
                try:
                    opt_state = torch.load(opt_last, map_location=device, weights_only=True)
                    opt_g.load_state_dict(opt_state['opt_g'])
                    scheduler_g.load_state_dict(opt_state['scheduler_g'])
                    if use_gan and 'opt_d' in opt_state:
                        opt_d.load_state_dict(opt_state['opt_d'])
                        if 'scheduler_d' in opt_state:
                            scheduler_d.load_state_dict(opt_state['scheduler_d'])
                    global_step = opt_state.get('global_step', 0)
                except Exception:
                    print("Warning: opt_last.pth corrupted, starting with fresh optimizer")

            history_path = os.path.join(exp_dir, 'history.json')
            if os.path.exists(history_path):
                try:
                    history = json.load(open(history_path))
                except Exception:
                    pass

            # Checkpoint may have saved epoch directly (newer runs) — use that.
            # Fall back to history length, then estimate from global_step.
            saved_epoch = locals().get('opt_state', {}).get('epoch', 0)
            if saved_epoch > 0:
                start_epoch = saved_epoch + 1
            elif len(history['train']['g_loss']) > 0:
                start_epoch = len(history['train']['g_loss']) + 1
            else:
                meta_path = os.path.join(exp_dir, 'meta.json')
                if os.path.exists(meta_path):
                    meta = json.load(open(meta_path))
                    saved_bs = meta.get('batch_size', train_config['batch_size'])
                    batches_per_epoch = 12800 // saved_bs
                else:
                    batches_per_epoch = len(train_loader)
                start_epoch = (global_step // batches_per_epoch) + 1

            print(f"Resuming from epoch {start_epoch}")
        else:
            print("No checkpoint found, starting from scratch")
            resume = False

    G.train()
    if use_gan:
        D.train()

    scaler = GradScaler('cuda')

    print(f"\nStarting training: {experiment_name}")
    print(f"  Epochs: {start_epoch}-{total_epochs} (total: {total_epochs - start_epoch + 1})")
    print(f"  Batch size: {train_config['batch_size']}, LR: {lr_g}, Lambda L1: {lambda_l1}")
    print(f"  GAN mode: {use_gan}, Resume: {resume}\n")

    for epoch in range(start_epoch, total_epochs + 1):
        epoch_g_loss = epoch_d_loss = epoch_l1_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs}", leave=False)

        for sar, eo in pbar:
            sar, eo = sar.to(device), eo.to(device)

            with autocast('cuda'):
                fake_eo = G(sar)
                if use_gan:
                    fake_logits = D(sar, fake_eo)
                    g_adv_loss = bce_loss(fake_logits, torch.ones_like(fake_logits))
                else:
                    g_adv_loss = torch.tensor(0.0, device=device)
                g_l1_loss = l1_loss(fake_eo, eo)
                g_loss = g_adv_loss + g_l1_loss * lambda_l1

            opt_g.zero_grad()
            scaler.scale(g_loss).backward()
            scaler.step(opt_g)

            if use_gan:
                with autocast('cuda'):
                    real_logits = D(sar, eo)
                    fake_logits = D(sar, fake_eo.detach())
                    d_loss = (bce_loss(real_logits, torch.ones_like(real_logits))
                            + bce_loss(fake_logits, torch.zeros_like(fake_logits))) * 0.5

                opt_d.zero_grad()
                scaler.scale(d_loss).backward()
                scaler.step(opt_d)
                epoch_d_loss += d_loss.item()

            scaler.update()
            epoch_g_loss += g_loss.item()
            epoch_l1_loss += g_l1_loss.item()
            num_batches += 1
            global_step += 1

            if global_step % log_interval == 0:
                pbar.set_postfix({
                    'G': f"{epoch_g_loss/num_batches:.4f}",
                    'D': f"{epoch_d_loss/num_batches:.4f}" if use_gan else 'N/A',
                    'L1': f"{epoch_l1_loss/num_batches:.4f}"
                })

        scheduler_g.step()
        if use_gan:
            scheduler_d.step()

        avg_g = epoch_g_loss / num_batches
        avg_d = epoch_d_loss / num_batches if use_gan else 0.0
        avg_l1 = epoch_l1_loss / num_batches

        history['train']['g_loss'].append(float(avg_g))
        history['train']['d_loss'].append(float(avg_d))
        history['train']['l1_loss'].append(float(avg_l1))

        if epoch % val_interval == 0 or epoch == total_epochs:
            G.eval()
            val_l1 = val_ssim = val_psnr = 0.0
            val_count = 0

            with torch.no_grad():
                for sar, eo in val_loader:
                    sar, eo = sar.to(device), eo.to(device)
                    with autocast('cuda'):
                        fake_eo = G(sar)
                        val_l1 += l1_loss(fake_eo, eo).item()
                    val_ssim += compute_ssim(fake_eo, eo)
                    val_psnr += compute_psnr(fake_eo, eo)
                    val_count += 1

            avg_val_l1 = val_l1 / val_count
            avg_val_ssim = val_ssim / val_count
            avg_val_psnr = val_psnr / val_count

            history['val']['l1_loss'].append(float(avg_val_l1))
            history['val']['ssim'].append(float(avg_val_ssim))
            history['val']['psnr'].append(float(avg_val_psnr))

            print(f"  [Epoch {epoch}] G={avg_g:.4f} D={avg_d:.4f} L1={avg_l1:.4f} | "
                  f"Val L1={avg_val_l1:.4f} SSIM={avg_val_ssim:.4f} PSNR={avg_val_psnr:.2f}")

            if avg_val_l1 < best_val_l1:
                best_val_l1 = avg_val_l1
                best_epoch = epoch
                torch.save(G.state_dict(), os.path.join(ckpt_dir, 'gen_best.pth'))
                if use_gan:
                    torch.save(D.state_dict(), os.path.join(ckpt_dir, 'disc_best.pth'))
                print(f"    New best: Val L1={best_val_l1:.4f}")

            G.train()
        else:
            print(f"  [Epoch {epoch}] G={avg_g:.4f} D={avg_d:.4f} L1={avg_l1:.4f}")

        if epoch % 10 == 0:
            torch.save(G.state_dict(), os.path.join(ckpt_dir, 'gen_last.pth'))
            if use_gan:
                torch.save(D.state_dict(), os.path.join(ckpt_dir, 'disc_last.pth'))
            opt_state = {
                'opt_g': opt_g.state_dict(),
                'scheduler_g': scheduler_g.state_dict(),
                'global_step': global_step,
                'epoch': epoch,
            }
            if use_gan:
                opt_state['opt_d'] = opt_d.state_dict()
                opt_state['scheduler_d'] = scheduler_d.state_dict()
            torch.save(opt_state, os.path.join(ckpt_dir, 'opt_last.pth'))

    torch.save(G.state_dict(), os.path.join(ckpt_dir, 'gen_last.pth'))
    if use_gan:
        torch.save(D.state_dict(), os.path.join(ckpt_dir, 'disc_last.pth'))
    torch.save({
        'opt_g': opt_g.state_dict(),
        'scheduler_g': scheduler_g.state_dict(),
        'global_step': global_step,
        'epoch': total_epochs,
        **({'opt_d': opt_d.state_dict(), 'scheduler_d': scheduler_d.state_dict()} if use_gan else {}),
    }, os.path.join(ckpt_dir, 'opt_last.pth'))

    json.dump(history, open(os.path.join(exp_dir, 'history.json'), 'w'), indent=2)
    plot_loss_curves(history, exp_dir, use_gan)

    G.load_state_dict(torch.load(os.path.join(ckpt_dir, 'gen_best.pth'),
                                  map_location=device, weights_only=True))
    G.eval()
    qual_dir = os.path.join(exp_dir, 'qualitative')
    os.makedirs(qual_dir, exist_ok=True)
    save_qualitative_triplets(G, test_loader, device, qual_dir, num_samples=10)

    print(f"\nTraining complete for {experiment_name}")
    print(f"  Best model: epoch {best_epoch} (Val L1={best_val_l1:.4f})")
    return history


def plot_loss_curves(history, exp_dir, use_gan):
    fig, axes = plt.subplots(1, 2 if use_gan else 1,
                             figsize=(12, 5) if use_gan else (6, 5))
    if not use_gan:
        axes = [axes]

    epochs = range(1, len(history['train']['g_loss']) + 1)

    axes[0].plot(epochs, history['train']['g_loss'], label='G loss', color='blue')
    axes[0].plot(epochs, history['train']['l1_loss'], label='L1 loss', color='orange', alpha=0.7)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Generator Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if use_gan:
        axes[1].plot(epochs, history['train']['d_loss'], label='D loss', color='red')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Discriminator Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, 'loss_curve.png'), dpi=150)
    plt.close()
    print(f"Loss curve saved to {os.path.join(exp_dir, 'loss_curve.png')}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--experiment', type=str, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    train(args.config, args.experiment, resume=args.resume)
