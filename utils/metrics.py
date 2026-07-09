import torch
import numpy as np
import os

try:
    import lpips as lpips_module
    _loss_fn = None
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: lpips not installed. Install with: pip install lpips")

try:
    from pytorch_fid.fid_score import calculate_fid_given_paths
    FID_AVAILABLE = True
except ImportError:
    FID_AVAILABLE = False

try:
    from skimage.metrics import structural_similarity as ssim_fn
    from skimage.metrics import peak_signal_noise_ratio as psnr_fn
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False


def compute_lpips(img1, img2, net='alex'):
    """Average LPIPS between two batches in [-1, 1]. Lower = more similar."""
    global _loss_fn
    if not LPIPS_AVAILABLE:
        return 0.0

    if _loss_fn is None:
        _loss_fn = lpips_module.LPIPS(net=net).to(img1.device)
    elif next(_loss_fn.parameters()).device != img1.device:
        _loss_fn = _loss_fn.to(img1.device)  # LPIPS defaults to CPU otherwise

    with torch.no_grad():
        dist = _loss_fn(img1, img2)
    return dist.mean().item()


def compute_ssim(img1, img2):
    """Average SSIM between two batches in [-1, 1]. Higher = more similar."""
    if not SKIMAGE_AVAILABLE:
        return 0.0

    img1_np = ((img1.detach().permute(0, 2, 3, 1).cpu().numpy()) + 1) / 2.0
    img2_np = ((img2.detach().permute(0, 2, 3, 1).cpu().numpy()) + 1) / 2.0

    scores = []
    for i in range(img1_np.shape[0]):
        score = ssim_fn(img1_np[i], img2_np[i], channel_axis=-1, data_range=1.0)
        scores.append(score)

    return np.mean(scores)


def compute_psnr(img1, img2):
    """Average PSNR between two batches in [-1, 1]. Higher = better."""
    if not SKIMAGE_AVAILABLE:
        return 0.0

    img1_np = ((img1.detach().permute(0, 2, 3, 1).cpu().numpy()) + 1) / 2.0
    img2_np = ((img2.detach().permute(0, 2, 3, 1).cpu().numpy()) + 1) / 2.0

    scores = []
    for i in range(img1_np.shape[0]):
        score = psnr_fn(img1_np[i], img2_np[i], data_range=1.0)
        scores.append(score)

    return np.mean(scores)


def compute_fid(pred_dir, gt_dir, batch_size=50, device='cuda'):
    """FID between two directories of images. Lower = more similar."""
    if FID_AVAILABLE:
        return calculate_fid_given_paths([pred_dir, gt_dir], batch_size=batch_size,
                                          device=device, dims=2048)
    else:
        return _compute_fid_manual(pred_dir, gt_dir, batch_size, device)


def _compute_fid_manual(pred_dir, gt_dir, batch_size=50, device='cuda'):
    """Manual FID using Inception v3 (fallback if pytorch-fid isn't installed)."""
    from torchvision.models import Inception_V3_Weights, inception_v3
    from torchvision import transforms as T

    inception = inception_v3(weights=Inception_V3_Weights.DEFAULT)
    inception.fc = torch.nn.Identity()
    inception.eval().to(device)

    transform = T.Compose([
        T.Resize((299, 299)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def get_features(image_dir):
        features = []
        files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])

        for i in range(0, len(files), batch_size):
            batch_files = files[i:i+batch_size]
            batch = []
            for f in batch_files:
                img = Image.open(os.path.join(image_dir, f)).convert('RGB')
                batch.append(transform(img))

            batch_tensor = torch.stack(batch).to(device)
            with torch.no_grad():
                feat = inception(batch_tensor)
            features.append(feat.cpu().numpy())

        return np.concatenate(features, axis=0)

    from PIL import Image

    mu1 = get_features(pred_dir)
    mu2 = get_features(gt_dir)

    sigma1 = np.cov(mu1, rowvar=False)
    sigma2 = np.cov(mu2, rowvar=False)

    from scipy.linalg import sqrtm
    diff = mu1.mean(0) - mu2.mean(0)
    cov_mean = sqrtm(sigma1 @ sigma2)

    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real

    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * cov_mean)
    return float(fid)
