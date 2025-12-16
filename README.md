# 3D Conditional WGAN-GP for Synthetic MRI Generation

**Project Status:** All unit tests pass (`4 passed`). The project is stable and ready for deployment.

##  Project Overview

This project implements a Conditional WGAN-GP (Wasserstein Generative Adversarial Network with Gradient Penalty) designed to synthesize high-resolution 3D volumetric MRI data. The model generates 3D volumes of size (64x64x64) conditioned on specific classes (e.g., disease state or tissue type).

The primary focus of this implementation is **computational stability** and **optimization for lower-resource environments (CPU)**, which is critical for 3D generative models.

##  Technical Rationale and Design Choices

The architecture was specifically designed to handle the challenges of 3D volumetric data and ensure reliable training on standard hardware:

| Component | Choice | Rationale |
| :--- | :--- | :--- |
| **GAN Type** | Conditional WGAN-GP | Provides superior **training stability** compared to standard GANs, eliminates mode collapse, and offers a more meaningful loss metric (Earth Mover's Distance). |
| **Normalization** | Instance Normalization (InstanceNorm3D) | Crucial for **Batch Size Stability**. InstanceNorm normalizes over individual samples, making the model robust and stable even when training with small batch sizes (e.g., Batch Size = 1) due to memory constraints in 3D, unlike Batch Normalization. |
| **Model Capacity** | CPUOptimized Models (`ngf`/`ndf`=32) | Models were kept intentionally lightweight (`ngf/ndf` starting at 32) to ensure low memory footprint and viability for **CPU-based training** and testing. |
| **Output Layer** | Single Channel Output (`C=1`) | The Generator's final layer is correctly set to output 1 channel (`nn.Tanh()`) to match the single-channel nature of MRI grayscale volumes. |

##  Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/YourUsername/3D_Conditional_GAN_MRI.git](https://github.com/YourUsername/3D_Conditional_GAN_MRI.git)
    cd 3D_Conditional_GAN_MRI
    ```

2.  **Install Dependencies:** Ensure you have Python 3.9+ and run:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Install Package in Editable Mode:** This is necessary to run unit tests and use the package structure correctly.
    ```bash
    pip install -e .
    ```

##  Usage

To start the training process, data should be organized into folders named after their class label (e.g., `0/`, `1/`, `2/`) under a root data directory.

```bash
python scripts/train_cli.py \
    --root-dir /path/to/your/mri_data_folder \
    --config-path config/default.yaml
    
