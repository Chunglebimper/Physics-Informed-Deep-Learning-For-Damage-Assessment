import torch
import numpy as np
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm


# Function to compute balanced class weights for loss calculation
def get_class_weights(dataset, weights_str):
    """

    :param dataset:
    :param weights_str: Manual balancing with multiplication; Should be a comma seperated list according to the following classes [0,1,2,3,4]
    :return:
    """
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

    # Break down weights_str to feed into manual weighting
    zero, one, two, three, four = weights_str.split(',')
    for cls in range(5):
        # classes      [  0,  1,  2,  3,  4]
        # earthquake = [  1,  1,  3, 10, 15] weights
        w = class_weight_dict.get(cls, 1.0)
        if cls == 0:
            w *= float(zero)
        elif cls == 1:
            w *= float(one)
        elif cls == 2:
            w *= float(two)
        elif cls == 3:
            w *= float(three)
        elif cls == 4:
            w *= float(four)
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
