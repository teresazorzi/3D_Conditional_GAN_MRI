import os
import time
import numpy as np
import nibabel as nib
import torch
import torch.optim as optim
from .models import CPUOptimizedGenerator3D, CPUOptimizedDiscriminator3D

def compute_gradient_penalty(D, real_samples, fake_samples, labels, device):
 """
    Calculates the gradient penalty for WGAN-GP. 
    
    This penalty ensures the discriminator satisfies the 1-Lipschitz constraint 
    by penalizing the norm of the gradients with respect to the input samples.
    
    :param D: Discriminator model.
    # ... (altri parametri e docstring) ...
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
    pass

def save_conditional_samples(G, epoch, num_classes, latent_dim, target_shape, 
                             device='cpu', samples_per_class=2):
    """
    Generates and saves a small batch of synthetic NIfTI samples for inspection 
    at a given epoch. This function includes dynamic Min-Max normalization for visualization.
    
    :param G: Generator model (must be in .eval() mode).
    # ... (altri parametri) ...
    """
    G.eval()
    os.makedirs('generated_samples', exist_ok=True)
    
    with torch.no_grad():
        for class_idx in range(num_classes):
            z = torch.randn(samples_per_class, latent_dim, device=device)
            labels = torch.full((samples_per_class,), class_idx, dtype=torch.long, device=device)
            fake_imgs = G(z, labels).cpu().numpy()
            
            for i in range(samples_per_class):
                img_3d = fake_imgs[i, 0]
                
                # Applichiamo la normalizzazione dinamica Min-Max per la visualizzazione
                # al posto della denormalizzazione fissa Tanh, dato che il modello 
                # tende a saturare l'output vicino a +1, anche se confinato.
                
                vmin = np.min(img_3d)
                vmax = np.max(img_3d)
                
                if vmax > vmin:
                    # Normalizzazione min-max dinamica: [min, max] -> [0, 1]
                    img_3d = (img_3d - vmin) / (vmax - vmin)
                else:
                     # Se l'immagine è uniforme (vmin=vmax), usa la denormalizzazione fissa 
                     # (utile solo se non ci sono dettagli, ma evita NaN)
                     img_3d = (img_3d + 1) / 2.0
                
                img_3d = np.clip(img_3d, 0, 1) # Assicura [0, 1]
                
                nifti_img = nib.Nifti1Image(img_3d.astype(np.float32), affine=np.eye(4))
                filename = f'generated_samples/epoch{epoch}_class{class_idx}_sample{i}.nii.gz'
                nib.save(nifti_img, filename)
    
    G.train()
    print(f"✓ Saved samples at resolution {target_shape}")
    pass

def train_conditional_gan(dataloader, num_classes, target_shape, num_epochs, 
                          latent_dim, lr, lambda_gp, n_critic, device, 
                          save_interval, ngf, ndf):
    """
    Executes the training loop for the Conditional WGAN-GP model.
    
    This function handles model initialization, optimizer setup, the alternating 
    D/G training updates, divergence control, and checkpoint saving.
    
    :param dataloader: PyTorch DataLoader providing MRI data.
    # ... (Docstring per tutti i parametri della configurazione)
    """
        # Soglie per la divergenza
    MAX_D_ABS_LOSS = 500.0  # Se |D_loss| > 500, c'è un'esplosione
    MAX_G_LOSS = 1500.0     # Se G_loss > 1500, c'è un'esplosione

    print(f"\n{'='*60}")
    print(f"CPU-OPTIMIZED GAN Training")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Target shape: {target_shape}")
    print(f"Number of classes: {num_classes}")
    print(f"Epochs: {num_epochs}")
    print(f"G Filters (ngf): {ngf}, D Filters (ndf): {ndf}")
    print(f"Latent Dim: {latent_dim}")
    print(f"G Learning Rate: {lr * 0.25}, D Learning Rate: {lr * 0.5}")
    print(f"Lambda GP: {lambda_gp}, N Critic: {n_critic}")
    print(f"Controllo divergenza: |D_loss| < {MAX_D_ABS_LOSS}, G_loss < {MAX_G_LOSS}")
    print(f"{'='*60}\n")
    
    # Initialize lightweight models using the passed ngf/ndf values
    G = CPUOptimizedGenerator3D(latent_dim=latent_dim, num_classes=num_classes, 
                                ngf=ngf, target_shape=target_shape).to(device)
    D = CPUOptimizedDiscriminator3D(num_classes=num_classes, ndf=ndf, 
                                    input_shape=target_shape).to(device)
    
    # Count parameters
    g_params = sum(p.numel() for p in G.parameters())
    d_params = sum(p.numel() for p in D.parameters())
    print(f"Generator parameters: {g_params:,} ({g_params/1e6:.2f}M)")
    print(f"Discriminator parameters: {d_params:,} ({d_params/1e6:.2f}M)")
    print(f"Total: {(g_params+d_params)/1e6:.2f}M parameters")
    print(f"Memory footprint: ~{(g_params+d_params)*4/1e9:.2f} GB\n")
    
    # Optimizers
    # LR_D = 0.000025, LR_G = 0.0000125
    optimizer_G = optim.Adam(G.parameters(), lr=lr * 0.25, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(D.parameters(), lr=lr * 0.5, betas=(0.5, 0.999))
    
    # Training metrics
    g_losses = []
    d_losses = []
    
    # Training loop with timing
    for epoch in range(num_epochs):
        epoch_start = time.time()
        epoch_d_loss = 0
        epoch_g_loss = 0
        n_batches = 0
        g_updates = 0
        
        for i, (real_imgs, labels) in enumerate(dataloader):
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)
            labels = labels.to(device)
            
            # Train Discriminator
            optimizer_D.zero_grad()
            
            z = torch.randn(batch_size, latent_dim, device=device)
            fake_imgs = G(z, labels).detach()
            
            real_validity = D(real_imgs, labels)
            fake_validity = D(fake_imgs, labels)
            
            gradient_penalty = compute_gradient_penalty(D, real_imgs, fake_imgs, labels, device)
            
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
            d_loss.backward()
            optimizer_D.step()
            
            epoch_d_loss += d_loss.item()
            
            # Train Generator
            if i % n_critic == 0:
                optimizer_G.zero_grad()
                
                z = torch.randn(batch_size, latent_dim, device=device)
                fake_imgs = G(z, labels)
                fake_validity = D(fake_imgs, labels)
                
                g_loss = -torch.mean(fake_validity)
                g_loss.backward()
                optimizer_G.step()
                
                epoch_g_loss += g_loss.item()
                g_updates += 1
            
            n_batches += 1
            
            # Print progress less frequently to save time
            if i % 5 == 0:
                elapsed = time.time() - epoch_start
                batches_per_sec = (i + 1) / elapsed if elapsed > 0 else 0
                eta_seconds = (len(dataloader) - i - 1) / batches_per_sec if batches_per_sec > 0 else 0
                eta_min = eta_seconds / 60
                
                print(f"[Epoch {epoch+1}/{num_epochs}] [Batch {i}/{len(dataloader)}] "
                      f"[D: {d_loss.item():.4f}] [G: {g_loss.item():.4f}] "
                      f"[Speed: {batches_per_sec:.2f} batch/s] [ETA: {eta_min:.1f}min]")
        
        avg_d_loss = epoch_d_loss / n_batches
        avg_g_loss = epoch_g_loss / max(g_updates, 1)
        
        d_losses.append(avg_d_loss)
        g_losses.append(avg_g_loss)
        
        epoch_time = time.time() - epoch_start
        print(f"\n>>> Epoch {epoch+1} Complete: D_loss={avg_d_loss:.4f}, G_loss={avg_g_loss:.4f}")
        print(f">>> Time: {epoch_time/60:.1f} minutes")
        print(f">>> Estimated total time: {epoch_time*num_epochs/3600:.1f} hours\n")

        # ============================================================
        # CONTROLLO DIVERGENZA (Early Stopping)
        # ============================================================
        if abs(avg_d_loss) > MAX_D_ABS_LOSS or avg_g_loss > MAX_G_LOSS:
             print("="*60)
             print(f"!!! DIVERGENZA RILEVATA: Le perdite sono esplose !!!")
             print(f"D_loss={avg_d_loss:.4f}, G_loss={avg_g_loss:.4f}")
             print(f"!!! Terminazione anticipata all'epoca {epoch+1} per instabilità. !!!")
             print("="*60)
             return G, D, g_losses, d_losses # Termina la funzione di training
        # ============================================================
        
        if (epoch + 1) % save_interval == 0:
            save_conditional_samples(G, epoch + 1, num_classes, latent_dim, target_shape, device)
            torch.save({
                'epoch': epoch + 1,
                'generator_state_dict': G.state_dict(),
                'discriminator_state_dict': D.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'g_losses': g_losses,
                'd_losses': d_losses,
                'num_classes': num_classes,
                'latent_dim': latent_dim,
                'target_shape': target_shape,
                'ngf': ngf, # <--- Parametro cruciale per la ricostruzione
                'ndf': ndf, # <--- Parametro cruciale per la ricostruzione
            }, f'checkpoint_epoch_{epoch+1}.pth')
            print(f"✓ Checkpoint saved\n")
    
    return G, D, g_losses, d_losses
    pass

def generate_class_specific_samples(generator_path, class_names,class_name, class_idx, num_samples=10, 
                                    latent_dim=64, num_classes=3, target_shape=(64,64,64), device='cpu'):
    """
    Loads a trained Generator from a checkpoint and generates a specified number 
    of synthetic samples for each class, saving them as NIfTI files. 
    
    :param generator_path: Path to the saved .pth checkpoint file.
    # ... (altri parametri) ...
    """
    print(f"\nGenerating {num_samples} samples for: {class_name}")
    print(f"Resolution: {target_shape}\n")
    
    checkpoint = torch.load(generator_path, map_location=device)
    
    # 1. Recupera i parametri dal checkpoint per ricostruire il modello
    if 'generator_state_dict' in checkpoint:
        saved_latent_dim = checkpoint.get('latent_dim', latent_dim)
        saved_num_classes = checkpoint.get('num_classes', num_classes)
        saved_target_shape = checkpoint.get('target_shape', target_shape)
        
        # 2. **CORREZIONE CRITICA**: Usa ngf=32 (o quello salvato) per ricostruire.
        ngf_for_reconstruction = checkpoint.get('ngf', 32)
        
        # Ricostruisci il modello con i parametri corretti
        G = CPUOptimizedGenerator3D(latent_dim=saved_latent_dim, num_classes=saved_num_classes, 
                                    ngf=ngf_for_reconstruction, target_shape=saved_target_shape).to(device)
        
        G.load_state_dict(checkpoint['generator_state_dict'])
    else:
        # Fallback se il checkpoint contiene solo lo state_dict (meno robusto)
        print("ATTENZIONE: Checkpoint non contiene i metadati, usando i parametri di default (ngf=32).")
        G = CPUOptimizedGenerator3D(latent_dim=latent_dim, num_classes=num_classes, 
                                    ngf=32, target_shape=target_shape).to(device)
        G.load_state_dict(checkpoint)
        
    G.eval()
    os.makedirs(f'generated_{class_name}', exist_ok=True)
    
    with torch.no_grad():
        z = torch.randn(num_samples, G.latent_dim, device=device) # Usa G.latent_dim ricostruito
        labels = torch.full((num_samples,), class_idx, dtype=torch.long, device=device)
        fake_imgs = G(z, labels).cpu().numpy()
    
    for i in range(num_samples):
        img_3d = fake_imgs[i, 0]
        
        # Applichiamo la normalizzazione dinamica Min-Max per la visualizzazione
        vmin = np.min(img_3d)
        vmax = np.max(img_3d)
        
        if vmax > vmin:
            # Normalizzazione min-max dinamica: [min, max] -> [0, 1]
            img_3d = (img_3d - vmin) / (vmax - vmin)
        else:
             # Se l'immagine è uniforme (vmin=vmax), usa la denormalizzazione standard 
             img_3d = (img_3d + 1) / 2.0
        
        img_3d = np.clip(img_3d, 0, 1)
        
        nifti_img = nib.Nifti1Image(img_3d.astype(np.float32), affine=np.eye(4))
        filename = f'generated_{class_name}/{class_name}_sample_{i:03d}.nii.gz'
        nib.save(nifti_img, filename)
    
    print(f"✓ Generated {num_samples} samples")
    pass
