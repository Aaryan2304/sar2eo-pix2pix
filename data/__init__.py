import os
import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np


def load_config(config_path):
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_paired_filename(sar_filename):
    """SAR filenames have _s1_, EO has _s2_. Swap the infix."""
    return sar_filename.replace('_s1_', '_s2_')


def create_splits(dataset_root, classes, split_ratios, seed=42):
    """Stratified train/val/test splits per terrain class.

    Shuffles within each class so adjacent patches don't leak between splits.
    """
    random.seed(seed)

    splits = {'train': [], 'val': [], 'test': []}

    for cls in classes:
        s1_dir = os.path.join(dataset_root, cls, 's1')
        sar_files = sorted(os.listdir(s1_dir))
        random.shuffle(sar_files)

        n = len(sar_files)
        n_train = int(n * split_ratios[0])
        n_val = int(n * split_ratios[1])

        splits['train'].extend(sar_files[:n_train])
        splits['val'].extend(sar_files[n_train:n_train + n_val])
        splits['test'].extend(sar_files[n_train + n_val:])

    return splits


def save_splits(splits, path):
    with open(path, 'w') as f:
        json.dump(splits, f, indent=2)


def load_splits(path):
    with open(path, 'r') as f:
        return json.load(f)


class SEN12Dataset(Dataset):
    """Paired SAR-EO dataset from SEN1-2.

    Returns (sar_tensor, eo_tensor) both in [-1, 1].
    """

    def __init__(self, dataset_root, split_files, classes, image_size=256):
        self.dataset_root = dataset_root
        self.split_files = split_files
        self.classes = classes
        self.image_size = image_size

        # Build filename → class mapping by scanning all class dirs.
        # Needed because split.json doesn't store which class a file belongs to.
        self.filename_to_class = {}
        for cls in classes:
            s1_dir = os.path.join(dataset_root, cls, 's1')
            if os.path.exists(s1_dir):
                for fname in os.listdir(s1_dir):
                    self.filename_to_class[fname] = cls

        # Only keep files that actually exist in the dataset
        self.pairs = []
        for fname in split_files:
            if fname in self.filename_to_class:
                self.pairs.append((self.filename_to_class[fname], fname))

        if len(self.pairs) < len(split_files):
            print(f"Warning: {len(split_files) - len(self.pairs)} files in split not found in dataset")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        cls, sar_fname = self.pairs[idx]
        eo_fname = get_paired_filename(sar_fname)

        sar_path = os.path.join(self.dataset_root, cls, 's1', sar_fname)
        eo_path = os.path.join(self.dataset_root, cls, 's2', eo_fname)

        # Load & resize (should already be 256x256, but just in case)
        sar_img = Image.open(sar_path).convert('L')
        eo_img = Image.open(eo_path).convert('RGB')
        sar_img = sar_img.resize((self.image_size, self.image_size), Image.BILINEAR)
        eo_img = eo_img.resize((self.image_size, self.image_size), Image.BILINEAR)

        # Normalise both to [-1, 1]
        sar_tensor = torch.from_numpy(np.array(sar_img, dtype=np.float32)) / 255.0
        sar_tensor = sar_tensor.unsqueeze(0)
        sar_tensor = (sar_tensor - 0.5) / 0.5

        eo_tensor = torch.from_numpy(np.array(eo_img, dtype=np.float32)) / 255.0
        eo_tensor = eo_tensor.permute(2, 0, 1)
        eo_tensor = (eo_tensor - 0.5) / 0.5

        return sar_tensor, eo_tensor


def denormalize(tensor):
    """[-1, 1] → [0, 1]"""
    return (tensor + 1.0) / 2.0


def tensor_to_pil(tensor):
    """Convert [-1, 1] tensor [C, H, W] to a PIL Image in [0, 255].

    Single-channel tensors get converted to grayscale then RGB.
    """
    tensor = denormalize(tensor.detach()).clamp(0, 1)
    arr = (tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    if arr.shape[2] == 1:
        arr = arr.squeeze(axis=2)
        return Image.fromarray(arr, mode='L').convert('RGB')
    return Image.fromarray(arr)


def get_dataloaders(config, seed=42):
    """Create train/val/test dataloaders from the config dict."""
    root = config['data']['root_dir']
    classes = config['data']['classes']
    ratios = config['data']['split_ratios']
    batch_size = config['training']['batch_size']
    num_workers = config['experiment']['num_workers']
    pin_memory = config['experiment']['pin_memory']

    split_path = os.path.join(root, 'splits.json')
    if os.path.exists(split_path):
        splits = load_splits(split_path)
    else:
        splits = create_splits(root, classes, ratios, seed)
        save_splits(splits, split_path)

    train_ds = SEN12Dataset(root, splits['train'], classes)
    val_ds = SEN12Dataset(root, splits['val'], classes)
    test_ds = SEN12Dataset(root, splits['test'], classes)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
        timeout=60  # avoid hanging on a corrupted image
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )

    print(f"Dataset split: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    return train_loader, val_loader, test_loader
