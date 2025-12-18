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
from mrisyngan.train import train_and_evaluate
from mrisyngan.utils import generate_class_specific_samples

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

    # 2. START TRAINING
    print("\nStarting GAN training...")

    # Use the reusable train_and_evaluate helper which loads defaults and applies overrides
    # We pass the loaded YAML config as overrides so command-line options are respected.
    score, model_state = train_and_evaluate(config, data_dir, device)

    # Save the final checkpoint
    final_checkpoint_path = 'cpu_gan_final.pth'
    torch.save({
        'generator_state_dict': model_state['generator_state_dict'],
        'discriminator_state_dict': model_state['discriminator_state_dict'],
        'g_losses': model_state.get('g_losses', []),
        # ... save all config/metadata needed for later reconstruction ...
        'latent_dim': model_state.get('latent_dim', config.get('LATENT_DIM')),
        'num_classes': model_state.get('num_classes'),
        'class_names': model_state.get('class_names'),
        'target_shape': model_state.get('target_shape', config.get('TARGET_SHAPE')),
        'ngf': model_state.get('ngf'), 
        'ndf': model_state.get('ndf')
    }, final_checkpoint_path)

    print(f"\nTRAINING COMPLETE! Final model saved: {final_checkpoint_path} | Score: {score:.4f}")

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