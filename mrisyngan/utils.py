# mrisyngan/utils.py

import os
import time
import numpy as np
import nibabel as nib
import torch
import torch.optim as optim

# Import models from your package (via __init__.py)
from .models import CPUOptimizedGenerator3D, CPUOptimizedDiscriminator3D


def compute_gradient_penalty(D, real_samples, fake_samples, labels, device, lambda_gp=10):
    """
    Calculates the gradient penalty for WGAN-GP. 
    """
    batch_size = real_samples.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, 1, device=device)
    
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = D(interpolates, labels)
    
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# --- (Other utility functions follow: save_conditional_samples, train_conditional_gan, etc.) ---
# These functions are stubs to complete the package structure needed for the tests.
# You will need to implement their full logic based on your training pipeline.

def save_conditional_samples(G, fixed_noise, fixed_labels, output_dir, epoch, target_shape=(64, 64, 64), device='cpu'):
    """Save generated 3D NIfTI samples."""
    G.eval()
    with torch.no_grad():
        samples = G(fixed_noise, fixed_labels).cpu()
    
    # Simple save logic for one sample per class
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for i in range(len(fixed_labels)):
        # Detach, clamp to [0, 1] if Tanh is used (or just [-1, 1] depends on normalization)
        # Here we just take the first channel and convert to numpy array
        volume_data = samples[i, 0].numpy()
        # Denormalize if necessary before saving, assuming normalization was [-1, 1]
        volume_data = (volume_data + 1) / 2 # Scale to [0, 1]
        
        nii_img = nib.Nifti1Image(volume_data, np.eye(4))
        label_val = fixed_labels[i].item()
        nib.save(nii_img, os.path.join(output_dir, f"epoch_{epoch}_class_{label_val}.nii.gz"))
    
    G.train()

def generate_class_specific_samples(G, num_samples, num_classes, latent_dim, target_shape, device='cpu'):
    """Generate a batch of samples, one for each class for visualization."""
    G.eval()
    
    # Create noise and labels for all classes
    noise = torch.randn(num_classes, latent_dim, device=device)
    labels = torch.arange(num_classes, device=device)
    
    with torch.no_grad():
        samples = G(noise, labels)
    
    G.train()
    return samples, labels


def train_conditional_gan(G, D, dataloader, config):
    """
    Main training function for the Conditional WGAN-GP.
    This function requires the full implementation of your training loop 
    (optimizers, scheduler, saving checkpoints, logging, etc.). 
    The implementation here is a simple placeholder to ensure testability.
    """
    # Placeholder for the actual training loop
    print("Starting GAN training (placeholder)...")
    
    device = torch.device(config['device'])
    G.to(device)
    D.to(device)
    
    # Example setup:
    optimizer_G = optim.Adam(G.parameters(), lr=config['lr_g'], betas=(0.5, 0.9))
    optimizer_D = optim.Adam(D.parameters(), lr=config['lr_d'], betas=(0.5, 0.9))
    
    start_time = time.time()
    
    for epoch in range(config['num_epochs']):
        for i, (real_images, labels) in enumerate(dataloader):
            
            real_images = real_images.to(device)
            labels = labels.to(device)
            batch_size = real_images.size(0)

            # ---------------------
            # Train Discriminator (D)
            # ---------------------
            D.zero_grad()
            
            # 1. Real samples
            d_real = D(real_images, labels)
            
            # 2. Fake samples
            z = torch.randn(batch_size, config['latent_dim'], device=device)
            fake_images = G(z, labels).detach()
            d_fake = D(fake_images, labels)
            
            # 3. Gradient Penalty
            gp = compute_gradient_penalty(D, real_images, fake_images, labels, device, config['lambda_gp'])
            
            # WGAN-GP Loss
            d_loss = -torch.mean(d_real) + torch.mean(d_fake) + gp * config['lambda_gp']
            
            d_loss.backward()
            optimizer_D.step()

            # ---------------------
            # Train Generator (G)
            # ---------------------
            if i % config['n_critic'] == 0:
                G.zero_grad()
                z = torch.randn(batch_size, config['latent_dim'], device=device)
                fake_labels = labels # Use the same labels for simplicity
                fake_images = G(z, fake_labels)
                
                # Discriminator output on fake samples
                d_fake_g = D(fake_images, fake_labels)
                
                # Generator Loss (Maximize D's fake score)
                g_loss = -torch.mean(d_fake_g)
                
                g_loss.backward()
                optimizer_G.step()
                
        # Simple logging
        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{config['num_epochs']}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")
            
    end_time = time.time()
    print(f"Training finished in {end_time - start_time:.2f} seconds.")