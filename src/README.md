# Damage Assessment Model

This is a modular PyTorch implementation for damage assessment using multi-temporal satellite imagery. The model supports physics-informed learning via GLCM-based texture loss, and includes tools for visualization, evaluation, and preprocessing.

## Data
The data used in this project is a reduced set from the xBD Damage Assessment dataset. The Ground Truth data has been converted from .json files to a map of pixels representing 1 of 5 classes. Furthermore, only the Earthquake data is being used. The data has been resized from 1024 x 1024 to 512 x 512 and cleaned to remove undesirable images.

## Features
- Uses transfer learning with ResNet50
- Multi-temporal dual-input ResNet+FPN backbone
- Sliding window patch extraction with priority sampling
- Class balancing via dynamic loss weighting
- Optional GLCM-based texture loss
- ROC and loss curve visualization
- xView2 scoring implementation


## TODO
- Put results in new directory
- Improve efficiency 
  - See if the default values from ImageNet are applicable to this data (mean and std)
  - Find appropriate stride and patch size
- More on stride:
  - The larger the stride the more the model notices broader features while a small stride captures finer detials
  - Noting the above, our data is filled with foliage so there is a lot of noise it is scanning. We really care about buildings.
- Patch Size
  - Bigger patches gives bigger context (consider patches as large as the biggest building or the smallest building)
- Understand: Dissect Loss file



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
python main.py --use_glcm --batch_size 4 --patch_size 64 --stride 32 --epochs 50 --data_root ../data
python main.py --use_glcm --batch_size 5 --patch_size 64 --stride 32 --epochs 50 --data_root ../data
python main.py --use_glcm --batch_size 6 --patch_size 64 --stride 32 --epochs 50 --data_root ../data
python main.py --use_glcm --batch_size 7 --patch_size 64 --stride 32 --epochs 50 --data_root ../data
python main.py --use_glcm --batch_size 8 --patch_size 64 --stride 32 --epochs 50 --data_root ../data
python main.py --use_glcm --batch_size 9 --patch_size 64 --stride 32 --epochs 50 --data_root ../data
python main.py --use_glcm --batch_size 10 --patch_size 64 --stride 32 --epochs 50 --data_root ../data
```

Disable GLCM loss with `--no-use_glcm`. Configure paths in `config.py` or pass `--data_root` at runtime.

## Authors
- Dr. Yan Lu, Brennan Miller, Abdul Anouti
