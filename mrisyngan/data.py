import os
import glob
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset # <-- QUESTA RIGA È CRITICA
from torchvision.transforms import functional as F
class MRINiftiDataset(Dataset):
    def __init__(self, main_dir, label, target_shape=TARGET_SHAPE, transform=None):
        self.paths = []
        self.labels = []
        self.transform = transform
        self.target_shape = target_shape
        
        for patient in sorted(os.listdir(main_dir)):
            patient_path = os.path.join(main_dir, patient)
            nifti_path = os.path.join(patient_path, "nifti")
            if not os.path.isdir(nifti_path):
                continue
            for fname in os.listdir(nifti_path):
                if fname.endswith(".nii") or fname.endswith(".nii.gz"):
                    file_path = os.path.join(nifti_path, fname)
                    try:
                        img = nib.load(file_path).get_fdata()
                        self.paths.append(file_path)
                        self.labels.append(label)
                    except Exception as e:
                        print(f"[✗] Errore: {file_path}: {e}")
                    break
        print(f"Trovati {len(self.paths)} file in {main_dir} (label={label})")
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        path = self.paths[idx]
        img = nib.load(path).get_fdata().astype(np.float32)
        
        # Normalizzazione: I dati sono ASSUNTI essere già normalizzati tra [-1, 1].
        # vmin, vmax = img.min(), img.max()
        # img = (img - vmin) / (vmax - vmin + 1e-6)
        # img = 2 * img - 1
        
        # Resize
        tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
        tensor = F.interpolate(
            tensor,
            size=self.target_shape,
            mode='trilinear',
            align_corners=False
        )
        tensor = tensor.squeeze(0)
        
        if self.transform:
            tensor = self.transform(tensor)
        return tensor, self.labels[idx]

def build_full_dataset(root_dir, target_shape=TARGET_SHAPE, transform=None):
    classes = sorted([d for d in os.listdir(root_dir) 
                      if os.path.isdir(os.path.join(root_dir, d))])
    print("Classi trovate:", classes)
    datasets = []
    for label, cls in enumerate(classes):
        class_path = os.path.join(root_dir, cls)
        datasets.append(MRINiftiDataset(class_path, label=label, 
                                       target_shape=target_shape, transform=transform))
    full_dataset = ConcatDataset(datasets)
    return full_dataset, classes
