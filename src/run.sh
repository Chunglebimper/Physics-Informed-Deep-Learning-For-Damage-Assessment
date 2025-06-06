#!/bin/bash

#python main.py --use_glcm --batch_size 4 --patch_size 64 --stride 32 --epochs 30 --data_root ../data

python main.py --batch_size 4 --patch_size 256 --stride 128 --epochs 1 --data_root ../data
