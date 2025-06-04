import argparse
import torch
from torch.utils.data import DataLoader
from dataset import DamageDataset
from model import EnhancedDamageModel
from metrics import compute_ordinal_conf_matrix, print_f1_per_class, calculate_xview2_score
from visuals import plot_multiclass_roc, visualize_predictions
from sklearn.metrics import accuracy_score, f1_score, precision_score
import numpy as np

def test_model(model_path, data_root, batch_size, patch_size, stride, device):
    # Prepare dataset
    dataset = DamageDataset(
        pre_dir=f"{data_root}/img_pre",
        post_dir=f"{data_root}/img_post",
        mask_dir=f"{data_root}/gt_post",
        patch_size=patch_size,
        stride=stride
    )
    loader = DataLoader(dataset, batch_size=batch_size)

    # Load model (resnet50)
    model = EnhancedDamageModel()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    y_true, y_pred, y_probs = [], [], []
    with torch.no_grad():
        for pre, post, mask, _ in loader:
            pre, post, mask = pre.to(device), post.to(device), mask.to(device)
            damage_out = model(pre, post)
            preds = torch.argmax(damage_out, dim=1)
            probs = torch.nn.functional.softmax(damage_out, dim=1).permute(0, 2, 3, 1).reshape(-1, 5)
            y_true.extend(mask.cpu().numpy().flatten())
            y_pred.extend(preds.cpu().numpy().flatten())
            y_probs.extend(probs.cpu().numpy())

    print("Confusion Matrix:\n", compute_ordinal_conf_matrix(y_true, y_pred))
    print_f1_per_class(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    xview2 = calculate_xview2_score(y_true, y_pred)
    precision_per_class = precision_score(y_true, y_pred, average=None, labels=range(5))
    macro_precision = precision_score(y_true, y_pred, average='macro')

    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"xView2 Score: {xview2:.4f}")
    for i, prec in enumerate(precision_per_class):
        print(f"Class {i} Precision: {prec:.4f}")
    print(f"Macro Precision: {macro_precision:.4f}")

    # Larger legend included
    plot_multiclass_roc(y_true, np.array(y_probs), n_classes=5)
    visualize_predictions(model, dataset, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the model")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--data_root', type=str, required=True, help='Path to dataset root')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--patch_size', type=int, default=128)
    parser.add_argument('--stride', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    test_model(args.model_path, args.data_root, args.batch_size, args.patch_size, args.stride, args.device)