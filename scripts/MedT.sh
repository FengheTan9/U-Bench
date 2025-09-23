#!/bin/bash


python main.py \
  --max_epochs 300 \
  --gpu 0 \
  --batch_size 8 \
  --model MedT \
  --base_dir ./data/tuscui \
  --dataset_name tuscui \
  --zero_shot_base_dir  ./data/TUCC-test/malignant \
  --zero_shot_dataset_name malignant

python main.py \
  --max_epochs 300 \
  --gpu 0 \
  --batch_size 8 \
  --model MedT \
  --base_dir ./data/isic18 \
  --dataset_name isic18 \
  --zero_shot_base_dir  ./data/PH2-test \
  --zero_shot_dataset_name PH2-test