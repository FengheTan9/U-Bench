#!/bin/bash


python main.py \
  --max_epochs 300 \
  --gpu 0 \
  --batch_size 8 \
  --model LFU_Net \
  --base_dir ./data/tuscui \
  --dataset_name tuscui \
  --zero_shot_base_dir  ./data/TUCC-test/malignant \
  --zero_shot_dataset_name malignant

python main.py \
  --max_epochs 300 \
  --gpu 0 \
  --batch_size 8 \
  --model LFU_Net \
  --base_dir ./data/isic18 \
  --dataset_name isic18 \
  --zero_shot_base_dir  ./data/PH2-test \
  --zero_shot_dataset_name PH2-test

python main.py \
  --max_epochs 300 \
  --gpu 0 \
  --batch_size 8 \
  --model LFU_Net \
  --base_dir ./data/BUSBRA \
  --dataset_name BUSBRA \
  --zero_shot_base_dir  ./data/bus \
  --zero_shot_dataset_name bus