# tests/test_models.py

import pytest
import torch
import torch.nn as nn
import numpy as np
import os
import tempfile
import yaml

# Importazione corretta dei moduli del progetto
from mrisyngan.models import CPUOptimizedGenerator3D, CPUOptimizedDiscriminator3D
from mrisyngan.utils import compute_gradient_penalty, train_conditional_gan, generate_class_specific_samples
from mrisyngan.data import MRINiftiDataset, build_full_dataset

# Configuration for testing (simplified)
TEST_CONFIG = {
    'latent_dim': 64,
    'num_classes': 3,
    'ngf': 32,
    'ndf': 32,
    'target_shape': [64, 64, 64],
    'device': 'cpu',
    'batch_size': 4,
    'test_batch_size': 2,
    'num_workers': 0,
    'num_epochs': 1,
    'lr_g': 0.0001,
    'lr_d': 0.0001,
    'n_critic': 1,
    'lambda_gp': 10.0,
}

# Fixture to create a temporary dummy NIfTI dataset for testing
@pytest.fixture(scope="session")
def dummy_data_dir():
    # Use tempfile.TemporaryDirectory to ensure cleanup
    with tempfile.TemporaryDirectory() as tmpdir:
        config = TEST_CONFIG
        num_classes = config['num_classes']
        target_shape = config['target_shape']
        
        # Create dummy data for each class
        for class_id in range(num_classes):
            class_path = os.path.join(tmpdir, str(class_id))
            os.makedirs(class_path, exist_ok=True)
            
            # Create a simple NIfTI file (e.g., a volume of ones)
            dummy_data = np.ones(target_shape, dtype=np.float32) * (class_id + 1)
            
            # Using nibabel to save a minimal NIfTI file
            nii_img = nib.Nifti1Image(dummy_data, np.eye(4))
            nib.save(nii_img, os.path.join(class_path, f'dummy_scan_{class_id}.nii.gz'))
            
            # Add one more file for good measure
            dummy_data = np.ones(target_shape, dtype=np.float32) * (class_id + 1) * 0.5
            nii_img = nib.Nifti1Image(dummy_data, np.eye(4))
            nib.save(nii_img, os.path.join(class_path, f'dummy_scan_{class_id}_2.nii.gz'))
            
        # Create a dummy config file
        with open(os.path.join(tmpdir, 'config.yaml'), 'w') as f:
            yaml.dump(config, f)
            
        yield tmpdir


# ====================================================================
# TEST 1: Model instantiation and shape verification (Bus Test)
# ====================================================================

def test_model_shapes():
    """
    Test 1: Checks if Generator and Discriminator initialize correctly 
    and produce the correct output shapes.
    """
    config = TEST_CONFIG
    latent_dim = config['latent_dim']
    num_classes = config['num_classes']
    target_shape = tuple(config['target_shape'])
    
    # Instantiate models
    G = CPUOptimizedGenerator3D(latent_dim, num_classes, target_shape=target_shape)
    D = CPUOptimizedDiscriminator3D(num_classes, input_shape=target_shape)

    batch_size = 4
    
    # 1. Generator Test: Input (Noise, Labels) -> Output (Image)
    z = torch.randn(batch_size, latent_dim)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    fake_images = G(z, labels)
    expected_shape = (batch_size, 1, *target_shape)
    
    assert fake_images.shape == expected_shape, f"G Output shape mismatch. Expected {expected_shape}, got {fake_images.shape}"
    assert fake_images.min() >= -1.0 and fake_images.max() <= 1.0, "G Output not bounded by Tanh [-1, 1]"
    
    # 2. Discriminator Test: Input (Image, Labels) -> Output (Score)
    real_images = torch.randn(batch_size, 1, *target_shape) # Dummy real image
    d_score = D(real_images, labels)
    
    expected_score_shape = (batch_size,)
    assert d_score.shape == expected_score_shape, f"D Output shape mismatch. Expected {expected_score_shape}, got {d_score.shape}"


# ====================================================================
# TEST 2: Data loading and DataLoader integrity
# ====================================================================

def test_data_loading(dummy_data_dir):
    """
    Test 2: Checks if the data pipeline loads the dummy data correctly 
    and returns correct tensor formats.
    """
    config = TEST_CONFIG
    dataloader = build_full_dataset(dummy_data_dir, config, is_test=True)
    
    assert len(dataloader.dataset) == 6, f"Expected 6 files, got {len(dataloader.dataset)}"
    
    # Check the first batch
    images, labels = next(iter(dataloader))
    
    batch_size = config['test_batch_size']
    target_shape = tuple(config['target_shape'])
    expected_img_shape = (batch_size, 1, *target_shape)
    
    assert images.shape == expected_img_shape, f"DataLoader image shape mismatch. Expected {expected_img_shape}, got {images.shape}"
    assert labels.shape == (batch_size,), f"DataLoader label shape mismatch. Expected ({batch_size},), got {labels.shape}"
    assert images.dtype == torch.float32, "Image data type should be float32"
    assert labels.dtype == torch.int64, "Label data type should be int64"
    assert images.min() >= -1.0 and images.max() <= 1.0, "Images should be normalized to [-1, 1]"


# ====================================================================
# TEST 3: Training utility functions check (GP and Full Train)
# ====================================================================

def test_gradient_penalty(dummy_data_dir):
    """
    Test 3a: Checks the gradient penalty calculation integrity.
    """
    config = TEST_CONFIG
    target_shape = tuple(config['target_shape'])
    
    # Setup
    D = CPUOptimizedDiscriminator3D(config['num_classes'], input_shape=target_shape).to(config['device'])
    batch_size = 4
    
    real_images = torch.randn(batch_size, 1, *target_shape, requires_grad=True)
    fake_images = torch.randn(batch_size, 1, *target_shape, requires_grad=True)
    labels = torch.randint(0, config['num_classes'], (batch_size,))
    
    # Test GP
    gp = compute_gradient_penalty(D, real_images, fake_images, labels, config['device'], config['lambda_gp'])
    
    assert isinstance(gp, torch.Tensor), "Gradient Penalty should be a torch.Tensor"
    assert gp.dim() == 0, "Gradient Penalty should be a scalar"
    assert not torch.isnan(gp).any(), "Gradient Penalty is NaN"

def test_train_loop_execution(dummy_data_dir, capsys):
    """
    Test 3b: Ensures the main training loop (placeholder) runs without crashing 
    for one epoch on dummy data.
    """
    config = TEST_CONFIG.copy()
    config['num_epochs'] = 1 # Only run 1 epoch for fast testing
    
    # Setup Models and Data
    G = CPUOptimizedGenerator3D(config['latent_dim'], config['num_classes'], target_shape=tuple(config['target_shape']))
    D = CPUOptimizedDiscriminator3D(config['num_classes'], input_shape=tuple(config['target_shape']))
    dataloader = build_full_dataset(dummy_data_dir, config, is_test=False)
    
    # Execute training (only checking for exceptions)
    try:
        train_conditional_gan(G, D, dataloader, config)
        assert True # Test passed if no exception raised
    except Exception as e:
        pytest.fail(f"Training loop crashed with exception: {e}")
    
    # Check if the print statements from the loop executed
    captured = capsys.readouterr()
    assert "Starting GAN training (placeholder)..." in captured.out