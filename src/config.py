import argparse
import torch

# This configuration file handles command-line arguments for flexible training/testing
def get_config():
    parser = argparse.ArgumentParser(description="Damage Assessment Model Configuration")

    # Enable GLCM-based texture loss for physics-informed learning
    parser.add_argument('--use_glcm', action='store_true', help='Enable GLCM-based texture loss')

    # Patch size for cropping satellite images (controls spatial granularity)
    parser.add_argument('--patch_size', type=int, default=128, help='Patch size for image patches')

    # Batch size for model training/testing
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for DataLoader')

    # Stride for sliding window to extract patches
    parser.add_argument('--stride', type=int, default=64, help='Stride for patch extraction')

    # Number of training epochs
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')

    # Learning rate for optimizer
    parser.add_argument('--lr', type=float, default=1e-6, help='Learning rate for optimizer')

    # Root directory containing img_pre, img_post, and gt_post
    parser.add_argument('--data_root', type=str, default='./data', help='Path to dataset root')

    # Device selection: CUDA if available, else CPU
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    return parser.parse_args()
