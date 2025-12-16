# mrisyngan/models.py

import torch
import torch.nn as nn
import torch.nn.functional as F 

# ============================================================
# VERY LIGHTWEIGHT GENERATOR - Optimized for CPU
# ============================================================
class CPUOptimizedGenerator3D(nn.Module):
    """
    VERY lightweight Generator for CPU. ngf=32 (reduced capacity for stability).
    Uses InstanceNorm3d and restored Tanh.
    """
    def __init__(self, latent_dim=64, num_classes=3, ngf=32, target_shape=(64, 64, 64)): 
        super(CPUOptimizedGenerator3D, self).__init__()
        
        # --- Base Assignments ---
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.target_shape = target_shape
        self.ngf = ngf # Save ngf as an attribute

        # Small embedding
        self.label_embedding = nn.Embedding(num_classes, latent_dim // 2)
        input_dim = latent_dim + latent_dim // 2
        
        # --- Layer Calculation Logic (Corrected) ---
        start_size = 4
        target_size = target_shape[0]
        num_layers = 0 
        current = start_size
        while current < target_size:
            current *= 2
            num_layers += 1
            
        # Initial Layer
        self.initial = nn.Sequential(
             nn.ConvTranspose3d(input_dim, ngf * 8, 4, 1, 0, bias=False), 
             nn.InstanceNorm3d(ngf * 8, affine=True), 
             nn.ReLU(True),
        )

        # Dynamic layers (Sequential Main)
        layers = []
        in_channels = ngf * 8

        for i in range(num_layers):
            is_final_upsample = (i == num_layers - 1)
            
            # CORRECTION: If it's the last layer, the output must be 1 (channel)
            out_channels = 1 if is_final_upsample else max(ngf, in_channels // 2) 
            
            layers.append(nn.ConvTranspose3d(in_channels, out_channels, 4, 2, 1, bias=False))
            
            if not is_final_upsample:
                layers.append(nn.InstanceNorm3d(out_channels, affine=True))
                layers.append(nn.ReLU(True))
            else:
                # Last layer: Tanh for output [-1, 1]
                layers.append(nn.Tanh()) 
                
            in_channels = out_channels
        
        self.main = nn.Sequential(*layers)

    def forward(self, z, labels):
        label_emb = self.label_embedding(labels)
        combined = torch.cat([z, label_emb], dim=1)
        # Expands for 3D Convolution
        combined = combined.view(combined.size(0), combined.size(1), 1, 1, 1) 
        
        x = self.initial(combined)
        x = self.main(x)
        
        # Ensure exact size
        if x.shape[2:] != self.target_shape:
             # Use F.interpolate from torch.nn.functional
             x = F.interpolate(x, size=self.target_shape, mode='trilinear', align_corners=False) 
        
        return x


# ============================================================
# VERY LIGHTWEIGHT DISCRIMINATOR - Optimized for CPU
# ============================================================
class CPUOptimizedDiscriminator3D(nn.Module):
    """
    VERY lightweight Discriminator for CPU. ndf=32.
    """
    def __init__(self, num_classes=3, ndf=32, input_shape=(64, 64, 64)): 
        super(CPUOptimizedDiscriminator3D, self).__init__()
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.ndf = ndf 
        
        # Small embedding
        self.label_embedding = nn.Embedding(num_classes, 32)
        spatial_size = input_shape[0] * input_shape[1] * input_shape[2]
        self.label_proj = nn.Linear(32, spatial_size)
        
        # Calculate how many layers are needed
        layers = []
        current_size = input_shape[0]
        in_channels = 2
        
        # First layer without InstanceNorm
        layers.append(nn.Conv3d(in_channels, ndf, 4, 2, 1, bias=False))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        in_channels = ndf
        current_size //= 2
        
        while current_size > 4:
            out_channels = min(ndf * 16, in_channels * 2)
            layers.append(nn.Conv3d(in_channels, out_channels, 4, 2, 1, bias=False))
            layers.append(nn.InstanceNorm3d(out_channels, affine=True)) 
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_channels = out_channels
            current_size //= 2
            
        # Final layer (Score)
        layers.append(nn.Conv3d(in_channels, 1, 4, 1, 0, bias=False))
        
        self.main = nn.Sequential(*layers)
    
    def forward(self, x, labels):
        batch_size = x.size(0)
        
        # Ensure correct size
        if x.shape[2:] != self.input_shape:
             x = F.interpolate(x, size=self.input_shape, mode='trilinear', align_corners=False)

        # BUG FIX: We force the image input to have a single channel (C=1).
        if x.dim() == 4:
             # Case (B, D, H, W). Add channel dimension (1).
             x = x.unsqueeze(1)
        elif x.dim() == 5 and x.size(1) != 1:
             # Case (B, C, D, H, W) where C > 1. Assume the first channel is the correct one.
             if x.size(1) > 1:
                 x = x[:, 0:1, :, :, :]
        
        label_emb = self.label_embedding(labels)
        label_emb = self.label_proj(label_emb)
        label_emb = label_emb.view(batch_size, 1, *self.input_shape)
        
        combined = torch.cat([x, label_emb], dim=1)
        output = self.main(combined)
        return output.view(-1)