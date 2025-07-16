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
    parser.add_argument('--verbose', action='store_true', help='Includes Epoch data in log')
    parser.add_argument('--sample_size', type=int,  default=128, help='Amount of images sampled')
    parser.add_argument('--levels', type=int, default=32, help='Levels in powers of two')
    parser.add_argument('--save_name', type=str, default='results')
    parser.add_argument('--weights_str', type=str, default='1,1,1,1,1')
    parser.add_argument('--class0and1percent', type=int, default=10)
    args = parser.parse_args()


    # ------ FUNCTION TO BE TIMED ------
    train_and_eval(
        use_glcm=args.use_glcm,
        patch_size=args.patch_size,
        stride=args.stride,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        root=args.data_root,
        verbose=args.verbose,
        sample_size=args.sample_size,
        levels=args.levels,
        save_name=args.save_name,
        weights_str=args.weights_str,
        class0and1percent=args.class0and1percent
    )
    # ---------------------------------
