import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, random_split
from dataset import DamageDataset
from model import EnhancedDamageModel
from loss import adaptive_texture_loss
from metrics import compute_ordinal_conf_matrix, print_f1_per_class, calculate_xview2_score
from utils import get_class_weights, analyze_class_distribution
from visuals import plot_loss_curves, plot_multiclass_roc, visualize_predictions
from sklearn.metrics import accuracy_score, f1_score, precision_score

# Function to train and evaluate the model
def train_and_eval(use_glcm, patch_size, stride, batch_size, epochs, lr, root):
    print(f"Training with texture loss: {use_glcm}")
    # Define data paths
    train_pre = os.path.join(root, "img_pre")
    train_post = os.path.join(root, "img_post")
    train_mask = os.path.join(root, "gt_post")

    # Load dataset with patching and stride
    dataset = DamageDataset(train_pre, train_post, train_mask, patch_size=patch_size, stride=stride)
    analyze_class_distribution(dataset)

    # Split into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Prepare DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize model, optimizer, loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnhancedDamageModel().to(device)
    weights = get_class_weights(train_dataset).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0
    best_macro_f1 = 0.0
    best_xview2 = 0.0
    train_loss_history = []
    val_loss_history = []

    # Training loop over epochs
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        model.train()
        if epoch < 3:
            # Freeze early layers to stabilize training
            for param in model.backbone.parameters():
                param.requires_grad = False
        else:
            for param in model.backbone.parameters():
                param.requires_grad = True

        total_loss = 0
        for pre, post, mask, _ in train_loader:
            pre, post, mask = pre.to(device), post.to(device), mask.to(device)
            optimizer.zero_grad()
            damage_out = model(pre, post)
            loss_ce = loss_fn(damage_out, mask)
            pred_classes = torch.argmax(damage_out, dim=1)
            if use_glcm:
                loss_tex = adaptive_texture_loss(pre, post, pred_classes)
                loss = loss_ce + 0.3 * loss_tex
            else:
                loss = loss_ce
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Log training loss
        train_loss = total_loss / len(train_loader)
        train_loss_history.append(train_loss)
        print(f"Train Loss: {train_loss:.4f}")

        # Validation phase
        model.eval()
        y_true, y_pred, y_probs = [], [], []
        val_loss = 0.0
        with torch.no_grad():
            for pre, post, mask, _ in val_loader:
                pre, post, mask = pre.to(device), post.to(device), mask.to(device)
                damage_out = model(pre, post)
                loss_ce = loss_fn(damage_out, mask)
                preds = torch.argmax(damage_out, dim=1)
                probs = F.softmax(damage_out, dim=1).permute(0, 2, 3, 1).reshape(-1, 5)
                y_true.extend(mask.cpu().numpy().flatten())
                y_pred.extend(preds.cpu().numpy().flatten())
                y_probs.extend(probs.cpu().numpy())
                val_loss += loss_ce.item()

        val_loss /= len(val_loader)
        val_loss_history.append(val_loss)

        # Metrics
        cm = compute_ordinal_conf_matrix(y_true, y_pred)
        print("Confusion Matrix:\n", cm)
        print_f1_per_class(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        xview2 = calculate_xview2_score(y_true, y_pred)
        print(f"Validation Accuracy: {acc:.4f}")
        print(f"Macro F1 Score: {macro_f1:.4f}")
        print(f"xView2 Score: {xview2:.4f}")

        # Track best scores and save model
        if xview2 > best_xview2:
            best_acc = acc
            best_macro_f1 = macro_f1
            best_xview2 = xview2
            torch.save(model.state_dict(), "best_model.pth")
            best_probs = np.array(y_probs)
            best_true = y_true
            best_preds = y_pred

    # Final metrics
    print("=== FINAL EVALUATION ===")
    print(f"Best Accuracy: {best_acc:.4f}")
    print(f"Best Macro F1: {best_macro_f1:.4f}")
    print(f"Best xView2 Score: {best_xview2:.4f}")

    # Precision calculation
    precision_per_class = precision_score(best_true, best_preds, average=None, labels=range(5))
    macro_precision = precision_score(best_true, best_preds, average='macro')
    print("=== FINAL PRECISION RESULTS ===")
    for i, prec in enumerate(precision_per_class):
        print(f"Class {i} Precision: {prec:.4f}")
    print(f"Macro Precision: {macro_precision:.4f}")

    # Visualization
    plot_multiclass_roc(best_true, best_probs, n_classes=5, class_names=[
        "Class 0: No Damage", "Class 1: Undamaged", "Class 2: Minor Damage",
        "Class 3: Major Damage", "Class 4: Destroyed"
    ])
    plot_loss_curves(train_loss_history, val_loss_history)
    visualize_predictions(model, val_dataset, device)
