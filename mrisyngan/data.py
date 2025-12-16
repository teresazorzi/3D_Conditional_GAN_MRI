# mrisyngan/data.py

import os
import glob
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset # Critical import
from torch.utils.data import DataLoader # Critical import
from torchvision.transforms import functional as F 

# Default constant defined locally
DEFAULT_TARGET_SHAPE = (64, 64, 64)

class MRINiftiDataset(Dataset):
    """
    Custom PyTorch Dataset for loading 3D MRI volumes stored as NIfTI files 
    and assigning a conditional label.
    """
    def __init__(self, main_dir, label, target_shape=DEFAULT_TARGET_SHAPE, transform=None):
        self.main_dir = main_dir
        self.label = label
        self.target_shape = target_shape
        self.transform = transform
        
        # Search for NIfTI files (e.g., .nii or .nii.gz) recursively
        search_pattern = os.path.join(self.main_dir, '**', '*.nii.gz')
        self.file_list = glob.glob(search_pattern, recursive=True)
        
        if not self.file_list:
            print(f"Warning: No NIfTI files found in {main_dir} with pattern {search_pattern}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        
        # Load NIfTI file
        nii_img = nib.load(file_path)
        data = nii_img.get_fdata()
        
        # --- SAFE NORMALIZATION (CORRECTION FOR NAN ERROR) ---
        
        # Convert to float32 immediately
        data = data.astype(np.float32)
        
        vmin = np.min(data)
        vmax = np.max(data)

        if vmax > vmin:
            # Scale to [0, 1]
            data = (data - vmin) / (vmax - vmin) 
        else:
            # If the image is uniform (vmax == vmin), set it to the normalized center (0). 
            # This prevents division by zero and NaN values.
            data = np.zeros_like(data) 

        # Scale to [-1, 1] (the required output range for the GAN)
        image = data * 2 - 1 

        # --- Tensor Conversion and Resizing ---

        # Convert to Pytorch Tensor
        # Add channel dimension (C=1) at index 0: (D, H, W) -> (1, D, H, W)
        image = torch.from_numpy(image).float().unsqueeze(0) 
        
        # Resize/Interpolate if needed
        current_shape = image.shape[1:]
        if current_shape != self.target_shape:
            # F.interpolate expects (N, C, D, H, W). We use unsqueeze(0) for batch dim.
            image = F.interpolate(image.unsqueeze(0), size=self.target_shape, mode='trilinear', align_corners=False).squeeze(0)
            
        if self.transform:
            image = self.transform(image)
            
        # The label is returned as a tensor
        label_tensor = torch.tensor(self.label, dtype=torch.long)
        
        return image, label_tensor

def build_full_dataset(base_data_path, config, is_test=False):
    """
    Combines datasets for different classes into a single DataLoader.
    
    Args:
        base_data_path (str): Root path where class folders (0, 1, 2) are located.
        config (dict): Configuration dictionary (must contain 'batch_size').
        is_test (bool): Flag to indicate if this is for testing.
        
    Returns:
        DataLoader: A combined DataLoader for all classes.
    """
    all_datasets = []
    num_classes = config['num_classes']

    # Iterate over class folders (0, 1, 2, ...)
    for label_id in range(num_classes):
        class_dir = os.path.join(base_data_path, str(label_id))
        
        if os.path.isdir(class_dir):
            dataset = MRINiftiDataset(
                main_dir=class_dir,
                label=label_id,
                target_shape=tuple(config['target_shape']),
                transform=None
            )
            # Filter out empty datasets (if no NIfTI files found)
            if len(dataset) > 0:
                all_datasets.append(dataset)
        else:
            print(f"Warning: Class directory not found: {class_dir}")

    if not all_datasets:
        raise FileNotFoundError("Error: No data found for any class. Check data path and structure.")

    # Combine all datasets
    combined_dataset = torch.utils.data.ConcatDataset(all_datasets)

    # Create DataLoader
    batch_size = config['test_batch_size'] if is_test else config['batch_size']
    
    dataloader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    return dataloader