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

def generate_class_specific_samples(generator_path, class_names, num_samples=10, 
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
