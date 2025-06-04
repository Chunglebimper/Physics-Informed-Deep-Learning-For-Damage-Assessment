# Damage Assessment Model

This is a modular PyTorch implementation for damage assessment using multi-temporal satellite imagery. The model supports physics-informed learning via GLCM-based texture loss, and includes tools for visualization, evaluation, and preprocessing.

## TODO
- Improve efficiency 
  - See if the default values from ImageNet are applicable to this data (mean and std)
  - Find appropriate stride and patch size
- More on stride:
  - The larger the stride the more the model notices broader features while a small stride captures finer detials
  - Noting the above, our data is filled with foliage so there is a lot of noise it is scanning. We really care about buildings.
- Patch Size
  - Bigger patches gives bigger context (consider patches as large as the biggest building or the smallest building)
- Understand: Dissect Loss file

## Features
- Multi-temporal dual-input ResNet+FPN backbone
- Sliding window patch extraction with priority sampling
- Class balancing via dynamic loss weighting
- Optional GLCM-based texture loss
- ROC and loss curve visualization
- xView2 scoring implementation

## Directory Structure

```
final_project/
    ├── data/
        ├── gt_post/
        ├── gt_pre/
        ├── img_post/
        ├── img_pre/
    ├── src/
        ├── config.py
        ├── dataset.py
        ├── loss.py
        ├── main.py
        ├── metrics.py
        ├── model.py
        ├── train.py
        ├── utils.py
        ├── visuals.py
        ├── README.md
```

## Usage

Given the above directory structure, use the following commands to train the model:

```bash
python main.py --use_glcm --batch_size 4 --patch_size 32 --stride 16 --epochs 30 --data_root ../data
python main.py --batch_size 4 --patch_size 32 --stride 16 --epochs 30  --data_root ../data
```

Disable GLCM loss with `--no-use_glcm`. Configure paths in `config.py` or pass `--data_root` at runtime.

## Authors
- Dr. Yan Lu, Brennan Miller, Abdul Anouti
