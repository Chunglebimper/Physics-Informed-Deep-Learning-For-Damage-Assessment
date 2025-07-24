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
        ├── log.py
        ├── loss.py
        ├── main.py
        ├── metrics.py
        ├── mkdir.py
        ├── model.py
        ├── train.py
        ├── utils.py
        ├── visuals.py
        ├── README.md
```

## Usage

Given the above directory structure, use the following baseline commands to train the model:

```bash
python main.py --use_glcm --batch_size 2 --patch_size 32 --stride 16 --epochs 50 --data_root ../data

```
### Required
* `--data_root` Path to dataset root directory

### Optional Commands
* `--use_glcm`: Disable GLCM loss by excluding it from runtime command
* `--patch_size`
* `--stride`     
* `--batch_size` 
* `--epochs` 
* `--lr` _Advanced testing_
* `--verbose`: Disable Epoch data in log by excluding it from runtime command
* `--sample_size ` _Advanced testing_
* `--levels`: _Advanced testing_
* `--save_name`: Send results to this directory name (appends to selected directory)
* `--weights_str`: Default is 'earthquake' (see `utils.py`)
* `--class0and1percent`: How much of class 0 and 1 to be included?
## Authors
- Dr. Yan Lu, Brennan Miller, Abdul Anouti, Caiden Pleis, Henry Stern
