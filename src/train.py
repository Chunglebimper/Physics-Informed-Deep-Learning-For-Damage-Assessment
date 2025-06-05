import numpy as np
import os
from log import Log
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from dataset import DamageDataset
from model import EnhancedDamageModel
from loss import adaptive_texture_loss
import metrics  # Import the entire module
from metrics import compute_ordinal_conf_matrix, calculate_xview2_score, print_f1_per_class, print_precision_per_class, print_recall_per_class
from utils import get_class_weights, analyze_class_distribution
from visuals import plot_loss_curves, plot_multiclass_roc, visualize_predictions, plot_epoch_accuracy
from sklearn.metrics import accuracy_score, f1_score, precision_score
import matplotlib.pyplot as plt

# Function to train and evaluate the model with logging

import metrics
print("Loaded metrics.py from:", metrics.__file__)

def train_and_eval(use_glcm, patch_size, stride, batch_size, epochs, lr, root):
    # create log file
    log = Log()
    log.open()
    params = use_glcm, patch_size, stride, batch_size, epochs, lr, root
    # begin logging
    log.append(f'{" Running config ":=^105}\n'
                f'{"".join(f"{str(i):<{15}}" for i in ("use_glcm", "patch_size", "stride", "batch_size", "epochs", "lr", "root"))}\n'
                f'{105 * "-"}\n'
                f'{"".join(f"{i:<{15}}" for i in (params))}\n'
                f'{105 * "-"}\n'
                f"{'Training on CUDA cores':<30}: {str(torch.cuda.is_available())}\n"
                f"{'Training with texture loss':<30}: {str(use_glcm)}\n"          
                f'{105 * "-"}'
                )
    print(f'Training on cuda cores: {torch.cuda.is_available()}')
    print(f"Training with texture loss: {use_glcm}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Prepare dataset paths
    train_pre = os.path.join(root, "img_pre")
    train_post = os.path.join(root, "img_post")
    train_mask = os.path.join(root, "gt_post")

    # Load dataset with patch size and stride
    dataset = DamageDataset(train_pre, train_post, train_mask, patch_size=patch_size, stride=stride)
    analyze_class_distribution(dataset)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize model, optimizer, and loss
    model = EnhancedDamageModel().to(device)
    print(f"Model loaded on: {next(model.parameters()).device}")
    weights = get_class_weights(train_dataset).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_acc, best_macro_f1, best_xview2 = 0.0, 0.0, 0.0
    train_loss_history, val_loss_history = [], []
    best_probs, best_true, best_preds = [], [], []

    # build accuracy graph over epoch time
    epochs_for_plotting = {}

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        start_time = time.perf_counter()  # Record the start time           PART OF TIME FUNCTION
        model.train()

        """
        changing the freezing layers could help output
        original 3
        """
        # Freeze backbone layers for first 3 epochs
        for param in model.backbone.parameters():
            param.requires_grad = epoch >= 3

        total_loss = 0
        for pre, post, mask, _ in train_loader:
            pre, post, mask = pre.to(device), post.to(device), mask.to(device)
            optimizer.zero_grad()
            damage_out = model(pre, post)
            loss_ce = loss_fn(damage_out, mask)
            pred_classes = torch.argmax(damage_out, dim=1)
            loss = loss_ce + 0.3 * adaptive_texture_loss(pre, post, pred_classes) if use_glcm else loss_ce
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

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

        cm = compute_ordinal_conf_matrix(y_true, y_pred)
        print("Confusion Matrix:\n", cm)
        print_f1_per_class(y_true, y_pred)

        ##############################################
        print_precision_per_class(y_true, y_pred)
        print_recall_per_class(y_true, y_pred)
        #############################################


        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        xview2 = calculate_xview2_score(y_true, y_pred)
        print(f"Validation Accuracy: {acc:.4f}")
        print(f"Macro F1 Score: {macro_f1:.4f}")
        print(f"xView2 Score: {xview2:.4f}")

        end_time = time.perf_counter()  # Record the end time                 PART OF TIME FUNCTION
        elapsed_time = end_time - start_time                                # PART OF TIME FUNCTION
        hours = int(elapsed_time // 3600)                                   # PART OF TIME FUNCTION
        minutes = int((elapsed_time % 3600) // 60)                          # PART OF TIME FUNCTION
        seconds = int(elapsed_time % 60)                                    # PART OF TIME FUNCTION
        print(f"Epoch {epoch+1} took: {hours: >2} hours, {minutes: >2} minutes, {seconds: >2} seconds")   # PART OF TIME FUNCTION
        #------------------------------
        epochs_for_plotting[(epoch+1)] = (acc)
        # ------------------------------


        # Track best scores and save model
        if xview2 > best_xview2:
            best_acc, best_macro_f1, best_xview2 = acc, macro_f1, xview2
            torch.save(model.state_dict(), "best_model.pth")
            best_probs, best_true, best_preds = np.array(y_probs), y_true.copy(), y_pred.copy()

        # ----------------------LOG-----------------------
        log.append(f"{'Epoch':<30}: {epoch + 1}/{epochs}")
        log.append(f"{'Train Loss':<30}: {train_loss:.4f}")
        log.append(f"{'Confusion Matrix':<30}:\n{cm}")
        log.append(f"{'Validation Accuracy':<30}: {acc:.4f}")
        log.append(f"{'Macro F1 Score':<30}: {macro_f1:.4f}")
        log.append(f"{'xView2 Score':<30}: {xview2:.4f}")
        log.append(f"{'Epoch Duration':<30}: {hours:>2} hours, {minutes:>2} minutes, {seconds:>2} seconds")
        if epoch + 1 < epochs:      # if all epochs printed, dont add seperator
            log.append("-" * 67)
        # ------------------------------------------------


    # Final metrics
    print("\n=== FINAL EVALUATION ===")

    print(f"Best Accuracy: {best_acc:.4f}")
    print(f"Best Macro F1: {best_macro_f1:.4f}")
    print(f"Best xView2 Score: {best_xview2:.4f}")
    #----------------------LOG-----------------------
    log.append(f"{' FINAL EVALUATION ':=^105}")
    log.append(f"{'Best Accuracy':<30}: {best_acc:.4f}")
    log.append(f"{'Best Macro F1: ':<30}: {best_macro_f1:.4f}")
    log.append(f"{'Best xView2 Score':<30}: {best_xview2:.4f}")
    #------------------------------------------------

    # Precision Calculation
    precision_per_class = precision_score(best_true, best_preds, average=None, labels=range(5), zero_division=0)
    macro_precision = precision_score(best_true, best_preds, average='macro', zero_division=0)
    print("=== FINAL PRECISION RESULTS ===")
    log.append(f"{' FINAL PRECISION RESULTS ':=^105}")                                      #LOG
    for i, prec in enumerate(precision_per_class):
        print(f"Class {i} Precision: {prec:.4f}")
        log.append(f"{f'Class {i} Precision':<30}: {prec:.4f}")                             #LOG
    print(f"Macro Precision: {macro_precision:.4f}")
    log.append(f"{f'CMacro Precision':<30}: {macro_precision:.4f}\n\n\n")                         #LOG


    # Visualizations
    plot_multiclass_roc(best_true, best_probs, n_classes=5, class_names=[
        "Class 0: No Damage", "Class 1: Undamaged", "Class 2: Minor Damage",
        "Class 3: Major Damage", "Class 4: Destroyed"
    ])
    plot_loss_curves(train_loss_history, val_loss_history)
    visualize_predictions(model, val_dataset, device)

    a = []
    count = 1
    print(epochs_for_plotting)
    for key_val_pair in epochs_for_plotting:
        a.append(epochs_for_plotting[(count)])
        count += 1
    #plot_epoch_accuracy(list(range(1, epochs)), a)
    plot_epoch_accuracy(range(0,epochs), a)

    log.close()                                                                            # BE SURE TO CLOSE LOG



