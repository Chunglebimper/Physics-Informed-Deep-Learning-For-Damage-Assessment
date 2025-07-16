import torch
import numpy as np
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

# Function to compute balanced class weights for loss calculation
def get_class_weights(dataset, weights_str):
    all_labels = []
    for i in range(len(dataset)):
        _, _, mask, _ = dataset[i]
        all_labels.extend(mask.flatten().numpy())
    labels_np = np.array(all_labels)
    present_classes = np.unique(labels_np)
    print(present_classes)

    # Calculate initial weights from sklearn
    weights = compute_class_weight('balanced', classes=present_classes, y=labels_np)
    class_weight_dict = dict(zip(present_classes, weights))
    print(class_weight_dict)

    # Apply manual scaling to focus on rare damage classes
    full_weights = []
    if weights_str == 'earthquake':
        print("Using earthquake weights...")
        for cls in range(5):
            w = class_weight_dict.get(cls, 1.0)
            if cls == 2:
                w *= 3.0
            elif cls == 3:
                w *= 10.0
            elif cls == 4:
                w *= 15.0
            full_weights.append(w)

    elif weights_str == 'flood':
        print("Using flood weights...")
        for cls in range(5):
            w = class_weight_dict.get(cls, 1.0)
            if cls == 2:
                w *= 2.0
            elif cls == 3:
                w *= 1.0
            elif cls == 4:
                w *= 2.0
            full_weights.append(w)

    elif weights_str == 'floodv2':
        print("Using floodv2 weights...")
        for cls in range(5):
            w = class_weight_dict.get(cls, 1.0)
            if cls == 2:
                w *= 3.0
            elif cls == 3:
                w *= 4.0
            elif cls == 4:
                w *= 5.0
            full_weights.append(w)

    else:
        print("USING DEFAULT 1:1 WEIGHTS")
        for cls in range(5):
            w = class_weight_dict.get(cls, 1.0)
            full_weights.append(w)

    print(f"Final class weights used in loss: {full_weights}")
    return torch.tensor(full_weights, dtype=torch.float32)

# Function to analyze and print dataset class distribution
def analyze_class_distribution(dataset, num_classes=5):
    print("Analyzing class distribution")
    counter = Counter()
    for i in tqdm(range(len(dataset)), desc="Scanning dataset"):
        _, _, mask, _ = dataset[i]
        counter.update(mask.flatten().tolist())
    total_pixels = sum(counter.values())
    for cls in range(num_classes):
        count = counter.get(cls, 0)
        percent = (count / total_pixels) * 100
        print(f"Class {cls}: {count:,} pixels ({percent:.2f}%)")
    print("-" * 50)
