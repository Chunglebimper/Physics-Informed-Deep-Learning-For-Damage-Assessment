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
from visuals import plot_loss_curves, plot_multiclass_roc, visualize_predictions, plot_epoch_accuracy, plot_epoch_f1
from sklearn.metrics import accuracy_score, f1_score, precision_score
import matplotlib.pyplot as plt
from mkdir import mkdir_results

# Function to train and evaluate the model with logging

import metrics
print("Loaded metrics.py from:", metrics.__file__)


def train_and_eval(use_glcm, patch_size, stride, batch_size, epochs, lr, root, verbose, sample_size, levels, save_name, weights_str, class0and1percent):
    TOTAL_start_time = time.perf_counter()  # Record the start time           PART OF TIME FUNCTION

    results_path = mkdir_results(save_name)                      # works to place contents of each individual run into respective directory
    os.makedirs(results_path, exist_ok=True)            # create output directory
    # create log file
    log = Log(path = f'{results_path}/log.txt')
    log.open()
    params1 = use_glcm, patch_size, stride, batch_size, epochs
    params2= lr, root, verbose, sample_size, levels
    # begin logging
    log.append(f'{" Running config ":=^110}\n'
                f'{"|".join(f"{str(i):^{14}}" for i in ("use_glcm", "patch_size", "stride", "batch_size", "epochs"))}\n'
                f'{70 * "-"}\n'
                f'{"|".join(f"{i:^{14}}" for i in params1)}\n'
                f'{70 * "="}\n'
                f'{"|".join(f"{str(i):^{14}}" for i in ("lr", "root", "verbose", "sample_size", "levels"))}\n'
                f'{70 * "-"}\n'
                f'{"|".join(f"{i:^{14}}" for i in params2)}\n'
                f'{110 * "-"}\n'
                f"{'Training on CUDA cores':<30}: {str(torch.cuda.is_available())}\n"
                f"{'Training with texture loss':<30}: {str(use_glcm)}\n"          
                f'{110 * "-"}'
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
    dataset = DamageDataset(train_pre, train_post, train_mask, class0and1percent=class0and1percent, patch_size=patch_size, stride=stride)
    print("Dataset loaded")
    analyze_class_distribution(dataset)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize model, optimizer, and loss
    model = EnhancedDamageModel().to(device)
    print(f"Model loaded on: {next(model.parameters()).device}")
    weights = get_class_weights(train_dataset, weights_str).to(device)
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
        changing the freezing layers could help output.
        original was 3
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
            loss = loss_ce + 0.3 * adaptive_texture_loss(pre, post, pred_classes, sample_size, levels) if use_glcm else loss_ce
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
        print_precision_per_class(y_true, y_pred)
        print_recall_per_class(y_true, y_pred)


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
        epochs_for_plotting[(epoch+1)] = (acc, macro_f1)
        # ------------------------------


        # Track best scores and save model
        if xview2 > best_xview2:
            best_acc, best_macro_f1, best_xview2 = acc, macro_f1, xview2
            torch.save(model.state_dict(), "best_model.pth")
            best_probs, best_true, best_preds = np.array(y_probs), y_true.copy(), y_pred.copy()

        # ----------------------LOG-----------------------
        if verbose:
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


    # Final metrics end of epoch loop
    print("\n=== FINAL EVALUATION ===")

    print(f"Best Accuracy: {best_acc:.4f}")
    print(f"Best Macro F1: {best_macro_f1:.4f}")
    print(f"Best xView2 Score: {best_xview2:.4f}")
    #----------------------LOG-----------------------
    log.append(f"{' FINAL EVALUATION ':=^110}")
    log.append(f"{'Best Accuracy':<30}: {best_acc:.4f}")
    log.append(f"{'Best Macro F1: ':<30}: {best_macro_f1:.4f}")
    log.append(f"{'Best xView2 Score':<30}: {best_xview2:.4f}")
    #------------------------------------------------

    # Precision Calculation
    precision_per_class = precision_score(best_true, best_preds, average=None, labels=range(5), zero_division=0)
    macro_precision = precision_score(best_true, best_preds, average='macro', zero_division=0)
    print("=== FINAL PRECISION RESULTS ===")
    log.append(f"{' FINAL PRECISION RESULTS ':=^110}")                                      #LOG
    for i, prec in enumerate(precision_per_class):
        print(f"Class {i} Precision: {prec:.4f}")
        log.append(f"{f'Class {i} Precision':<30}: {prec:.4f}")                             #LOG
    print(f"Macro Precision: {macro_precision:.4f}")
    log.append(f"{f'CMacro Precision':<30}: {macro_precision:.4f}\n\n\n")                         #LOG


    # Visualizations
    plot_multiclass_roc(best_true, best_probs, n_classes=5, class_names=[
        "Class 0: No Damage", "Class 1: Undamaged", "Class 2: Minor Damage",
        "Class 3: Major Damage", "Class 4: Destroyed"],
        save_path=f'{results_path}/plot_multiclass_roc.jpg')
    plot_loss_curves(train_loss_history, val_loss_history, save_path=f'{results_path}/plot_loss_curves.jpg')
    visualize_predictions(model, val_dataset, device, save_path=f'{results_path}/visualize_predictions.jpg')


    acc_list = []
    f1_list = []
    total_acc = 0
    total_f1 = 0
    count = 1
    for key_val_pair in epochs_for_plotting:
        acc_list.append((epochs_for_plotting[(count)])[0])
        f1_list.append((epochs_for_plotting[(count)])[1])
        count += 1

    # Calculations
    for i in acc_list:
        total_acc = i + total_acc
    for i in f1_list:
        total_f1 = i + total_f1
    avg_accuracy = total_acc / len(acc_list)
    avg_f1 = total_f1 / len(f1_list)
    avg_valoss = sum(val_loss_history) / len(val_loss_history)


    log.append(f"{'Average Accuracy':<30}: {avg_accuracy:.4f}")
    log.append(f"{'Average f1':<30}: {avg_f1:.4f}")
    plot_epoch_accuracy(range(0,epochs), acc_list, save_path=f'{results_path}/plot_epoch_accuracy.jpg')
    plot_epoch_f1(range(0, epochs), f1_list, save_path=f'{results_path}/plot_epoch_f1.jpg')

    # End timing function
    TOTAL_end_time = time.perf_counter()  # Record the end time                  PART OF TIME FUNCTION
    TOTAL_elapsed_time = TOTAL_end_time - TOTAL_start_time                                 # PART OF TIME FUNCTION
    TOTAL_hours = int(TOTAL_elapsed_time // 3600)                                    # PART OF TIME FUNCTION
    TOTAL_minutes = int((TOTAL_elapsed_time % 3600) // 60)                           # PART OF TIME FUNCTION
    TOTAL_seconds = int(TOTAL_elapsed_time % 60)                                     # PART OF TIME FUNCTION
    log.append(f'Seconds elapsed: {TOTAL_elapsed_time}')
    log.append(f"Total elapsed time: {TOTAL_hours: >6} hours, {TOTAL_minutes: >6} minutes, {TOTAL_seconds: >6} seconds")
    log.append(f"{TOTAL_elapsed_time}, {avg_f1}, {macro_precision}, {avg_accuracy}, {avg_valoss}")
    log.append("TOTAL_elapsed_time, avg_f1, macro_precision, avg_accuracy, avg_valoss")
    log.close()                                                          # BE SURE TO CLOSE LOG
    return best_macro_f1, avg_accuracy, avg_valoss