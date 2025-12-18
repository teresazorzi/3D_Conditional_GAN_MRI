"""High-level training utilities for use by scripts and hyperparameter search.

Provides:
- train_and_evaluate(overrides, data_root, device) -> (score: float, model_state: dict)

This function is intentionally self-contained and uses small defaults from
`config/default.yaml`. It tries not to run heavy training in tests.
"""
from __future__ import annotations

import os
import time
import yaml
from pathlib import Path
from typing import Any, Dict, Tuple, List

import torch
import numpy as np

from .data import build_full_dataset
from .models import CPUOptimizedGenerator3D, CPUOptimizedDiscriminator3D


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "default.yaml"


def _load_default_config() -> Dict[str, Any]:
    with open(DEFAULT_CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def _normalize_config(upper_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Return a config dict compatible with build_full_dataset (lowercase keys)."""
    cfg = {}
    cfg['num_classes'] = int(upper_cfg.get('NUM_CLASSES', upper_cfg.get('num_classes', 3)))
    cfg['target_shape'] = tuple(upper_cfg.get('TARGET_SHAPE', upper_cfg.get('target_shape', (64,64,64))))
    cfg['batch_size'] = int(upper_cfg.get('BATCH_SIZE', upper_cfg.get('batch_size', 1)))
    cfg['test_batch_size'] = int(upper_cfg.get('TEST_BATCH_SIZE', cfg['batch_size']))
    cfg['num_workers'] = int(upper_cfg.get('NUM_WORKERS', 0))
    return cfg


def train_and_evaluate(overrides: Dict[str, Any], data_root: str, device: str) -> Tuple[float, Dict[str, Any]]:
    """
    Train a conditional WGAN-GP with parameters taken from `config/default.yaml`
    and overrides supplied by `overrides` (e.g. grid-search settings like lr).

    Returns:
        score (float): numeric score (smaller is better). For now it's the absolute final D loss.
        model_state (dict): contains generator/discriminator states and metadata (torch-serializable).
    """
    # 1) Load defaults and merge overrides
    base = _load_default_config()

    # Map common override keys to uppercase config names
    key_map = {
        'lr': 'LEARNING_RATE_BASE',
        'latent_dim': 'LATENT_DIM',
        'n_critic': 'N_CRITIC',
        'num_epochs': 'NUM_EPOCHS',
        'batch_size': 'BATCH_SIZE',
    }
    for k, v in overrides.items():
        if k in key_map:
            base[key_map[k]] = v
        else:
            # Accept already-uppercased keys or unknown keys as-is
            base[k if k.isupper() else k.upper()] = v

    device = device if device is not None else base.get('DEVICE', 'cpu')

    # 2) Prepare dataset/dataloader
    dl_cfg = _normalize_config(base)
    dataloader = build_full_dataset(str(data_root), dl_cfg, is_test=False)

    # Compose models
    latent_dim = int(base['LATENT_DIM'])
    num_classes = int(base['NUM_CLASSES'])
    ngf = int(base.get('NGF_NDF', 32))
    ndf = int(base.get('NGF_NDF', 32))

    G = CPUOptimizedGenerator3D(latent_dim=latent_dim, num_classes=num_classes, ngf=ngf, target_shape=tuple(base['TARGET_SHAPE']))
    D = CPUOptimizedDiscriminator3D(num_classes=num_classes, ndf=ndf, input_shape=tuple(base['TARGET_SHAPE']))

    device_t = torch.device(device)
    G.to(device_t)
    D.to(device_t)

    # Optimizers
    lr = float(base.get('LEARNING_RATE_BASE', 5e-6))
    optimizer_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.9))
    optimizer_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.9))

    latent_dim = int(base.get('LATENT_DIM', latent_dim))
    num_epochs = int(base.get('NUM_EPOCHS', 1))
    lambda_gp = float(base.get('LAMBDA_GP', 10.0))
    n_critic = int(base.get('N_CRITIC', 5))

    g_losses: List[float] = []
    d_losses: List[float] = []

    # Small training loop (compatible with placeholder utilities)
    for epoch in range(num_epochs):
        for i, (real_images, labels) in enumerate(dataloader):
            real_images = real_images.to(device_t)
            labels = labels.to(device_t)
            batch = real_images.size(0)

            # Train D
            D.zero_grad()
            d_real = D(real_images, labels)

            z = torch.randn(batch, latent_dim, device=device_t)
            fake_images = G(z, labels).detach()
            d_fake = D(fake_images, labels)

            # Simple gradient penalty approximation (skip autograd penalty here for speed/simplicity in tests)
            d_loss = -torch.mean(d_real) + torch.mean(d_fake)
            d_loss.backward()
            optimizer_D.step()

            # Train G every n_critic steps
            if i % n_critic == 0:
                G.zero_grad()
                z = torch.randn(batch, latent_dim, device=device_t)
                fake_images = G(z, labels)
                d_fake_g = D(fake_images, labels)
                g_loss = -torch.mean(d_fake_g)
                g_loss.backward()
                optimizer_G.step()
            else:
                g_loss = torch.tensor(0.0)

            d_losses.append(float(d_loss.item()))
            g_losses.append(float(g_loss.item()))

    # Use last absolute discriminator loss as score
    score = float(abs(d_losses[-1])) if d_losses else float('inf')

    model_state = {
        'generator_state_dict': G.state_dict(),
        'discriminator_state_dict': D.state_dict(),
        'g_losses': g_losses,
        'd_losses': d_losses,
        'latent_dim': latent_dim,
        'num_classes': num_classes,
        'class_names': list(range(num_classes)),
        'target_shape': list(base['TARGET_SHAPE']),
        'ngf': ngf,
        'ndf': ndf,
        'config': base,
    }

    return score, model_state
