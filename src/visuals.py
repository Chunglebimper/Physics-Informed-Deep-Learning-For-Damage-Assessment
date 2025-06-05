import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, accuracy_score


# Function to plot the ROC curve for multiple classes
def plot_multiclass_roc(y_true, y_prob, n_classes, class_names=None, save_path='../results/plot_multiclass_roc.jpg'):

    # Convert true labels to binary format for ROC calculation
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
    fpr = dict(); tpr = dict(); roc_auc = dict()

    # Calculate ROC curve and AUC for each class
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        label = f"Class {i}" if not class_names else class_names[i]
        plt.plot(fpr[i], tpr[i], lw=2, label=f'{label} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('Multi-Class ROC Curve', fontsize=20)
    plt.legend(loc='lower right', fontsize=16, title="Classes", title_fontsize=18)
    plt.grid(True)
    plt.tight_layout()

    # Save to file if path provided
    if save_path:
        plt.savefig(save_path)
        print(f"ROC curve saved to {save_path}")

    #plt.show()

# === Function to plot the training and validation loss curves ===
def plot_loss_curves(train_loss_history, val_loss_history, save_path='../results/plot_loss_curves.jpg'):
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss_history, label="Training Loss", linewidth=2)
    plt.plot(val_loss_history, label="Validation Loss", linewidth=2)
    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel("Loss", fontsize=16)
    plt.title("Training vs Validation Loss Curves", fontsize=20)
    plt.legend(fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    #plt.show()

    # Save to file if path provided
    if save_path:
        plt.savefig(save_path)
        print(f"ROC curve saved to {save_path}")


# Function to visualize sample predictions
def visualize_predictions(model, dataset, device, num_samples=3, save_path='../results/visualize_predictions.jpg'):
    import numpy as np
    model.eval()
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    for idx in indices:
        pre, post, mask, name = dataset[idx]
        pre = pre.unsqueeze(0).to(device)
        post = post.unsqueeze(0).to(device)
        with torch.no_grad():
            damage_out = model(pre, post)
            pred = torch.argmax(damage_out.squeeze(), dim=0).cpu().numpy()
            unique, counts = np.unique(pred, return_counts=True)
            print("Prediction distribution:", dict(zip(unique, counts)))

        # Convert image tensors back to displayable format
        post_img = post.squeeze().permute(1, 2, 0).cpu().numpy()
        post_img = (post_img * 0.229 + 0.456) * 255
        post_img = np.clip(post_img, 0, 255).astype(np.uint8)

        # Plot original post-disaster image, ground truth mask, and prediction
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(post_img)
        axs[1].imshow(mask.cpu(), vmin=0, vmax=4, cmap='jet')
        axs[2].imshow(pred, vmin=0, vmax=4, cmap='jet')
        for ax in axs: ax.axis('off')
        plt.suptitle(f"Sample: {name}", fontsize=16)
        plt.tight_layout()
        #plt.show()

        # Save to file if path provided
        if save_path:
            plt.savefig(save_path)
            print(f"ROC curve saved to {save_path}")


def plot_epoch_accuracy(epochs, accuracy, save_path='../results/plot_epoch_accuracy.jpg'):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, accuracy, linewidth=2)
    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel("Accuracy", fontsize=16)
    plt.title("Accuracy per Epoch", fontsize=20)
    plt.legend(fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    # plt.show()

    # Save to file if path provided
    if save_path:
        plt.savefig(save_path)
        print(f"ROC curve saved to {save_path}")