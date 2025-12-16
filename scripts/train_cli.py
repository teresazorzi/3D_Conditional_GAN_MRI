"""
Command Line Interface (CLI) script for training the 3D Conditional WGAN-GP.

This script manages external configuration loading (YAML) and argument parsing, 
ensuring the model training is reproducible and easy to execute.
"""
import argparse
import yaml
import torch
import os
import sys
from torch.utils.data import DataLoader

# Import your modules
# NOTE: Ensure you have implemented these classes/functions in the respective files
from mrisyngan.data import build_full_dataset
from mrisyngan.utils import train_conditional_gan, generate_class_specific_samples

def load_config(config_path):
    """Loads configuration parameters from a YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"ERROR: Configuration file not found at {config_path}")
        sys.exit(1)

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train the 3D Conditional WGAN-GP model for MRI synthesis."
    )
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/default.yaml', 
        help='Path to the YAML configuration file. (Default: config/default.yaml)'
    )
    # The --root-dir argument allows users to easily override the data path without editing the config file.
    parser.add_argument(
        '--root-dir', 
        type=str, 
        required=False, 
        help='[OVERRIDE] Root directory for the NIfTI dataset. Overrides the value in the config file.'
    )
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_config(args.config)
    
    # 1. ROOT DIRECTORY VALIDATION (Prerequisite Check)
    # The CLI argument takes priority over the YAML file.
    if args.root_dir:
        config['ROOT_DIR'] = args.root_dir
    
    data_dir = config.get('ROOT_DIR')
    if data_dir is None or not os.path.isdir(data_dir):
        # Use an explicit Exception/Exit as per exam guidelines (Don't use assert for important conditions)
        print("="*60)
        print(f"ERROR: Data directory not found or not specified.")
        print(f"Please check ROOT_DIR in {args.config} or use --root-dir argument.")
        print("="*60)
        sys.exit(1)

    print("--- GAN Training Setup ---")
    print(f"Data Directory: {data_dir}")
    print(f"Device: {config.get('DEVICE', 'cpu')}")
    print(f"Target Shape: {config.get('TARGET_SHAPE')}\n")

    # 2. DATA LOAD (Using mrisyngan/data.py)
    print("Loading dataset...")
    target_shape = tuple(config['TARGET_SHAPE'])
    device = torch.device(config.get('DEVICE', 'cpu'))

    try:
        dataset, class_names = build_full_dataset(data_dir, target_shape=target_shape)
    except Exception as e:
        # Catch errors during NIfTI loading (e.g., bad file format)
        raise RuntimeError(f"Failed to build dataset from {data_dir}: {e}") from e

    print(f"Total samples: {len(dataset)}")
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    train_loader = DataLoader(
        dataset, 
        batch_size=config['BATCH_SIZE'],
        shuffle=True, 
        num_workers=config.get('NUM_WORKERS', 0),
        pin_memory=False
    )
    
    # 3. START TRAINING (The "Simulation" Step)
    print("\nStarting GAN training...")
    G_final, D_final, g_losses, d_losses = train_conditional_gan(
        dataloader=train_loader,
        num_classes=len(class_names),
        target_shape=target_shape,
        num_epochs=config['NUM_EPOCHS'],
        latent_dim=config['LATENT_DIM'],
        lr=config['LEARNING_RATE_BASE'], # Critical stability parameter
        lambda_gp=config['LAMBDA_GP'],
        n_critic=config['N_CRITIC'],
        device=device,
        save_interval=config['SAVE_INTERVAL'],
        ngf=config['NGF_NDF'],
        ndf=config['NGF_NDF']
    )
    
    # Save the final checkpoint
    final_checkpoint_path = 'cpu_gan_final.pth'
    torch.save({
        'generator_state_dict': G_final.state_dict(),
        'discriminator_state_dict': D_final.state_dict(),
        'g_losses': g_losses,
        # ... save all config/metadata needed for later reconstruction ...
        'latent_dim': config['LATENT_DIM'],
        'num_classes': len(class_names),
        'class_names': class_names,
        'target_shape': config['TARGET_SHAPE'],
        'ngf': config['NGF_NDF'], 
        'ndf': config['NGF_NDF']
    }, final_checkpoint_path)
    print(f"\nTRAINING COMPLETE! Final model saved: {final_checkpoint_path}")

    # 4. GENERATE SAMPLES (The "Analysis" Step, separated from simulation)
    print("\nStarting final sample generation (Analysis phase)...")
    for class_name, class_idx in class_to_idx.items():
        generate_class_specific_samples(
            generator_path=final_checkpoint_path,
            class_idx=class_idx,
            class_name=class_name,
            num_samples=10,
            latent_dim=config['LATENT_DIM'],
            num_classes=len(class_names),
            target_shape=target_shape,
            device=device
        )
    
    print("\n--- ALL DONE! ---")

if __name__ == '__main__':
    main()