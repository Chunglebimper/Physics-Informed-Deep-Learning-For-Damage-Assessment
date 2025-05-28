import argparse
from train import train_and_eval

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Damage Assessment Training")
    parser.add_argument('--use_glcm', action='store_true', help='Enable GLCM texture loss')
    parser.add_argument('--patch_size', type=int, default=128, help='Patch size for image patches')
    parser.add_argument('--stride', type=int, default=64, help='Stride for patch extraction')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-6, help='Learning rate')
    parser.add_argument('--data_root', type=str, required=True, help='Path to dataset root directory')
    args = parser.parse_args()

    train_and_eval(
        use_glcm=args.use_glcm,
        patch_size=args.patch_size,
        stride=args.stride,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        root=args.data_root
    )
