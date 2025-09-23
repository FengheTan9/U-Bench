#!/bin/bash


python main_multi3d.py \
  --max_epochs 300 \
  --gpu 1 \
  --batch_size 8 \
  --model PraNet \
  --base_dir ./data/synapse \
  --dataset_name synapse \
  --num_classes 9 \
  --input_channel 3 \
  --val_interval 100


python main_multi3d.py \
  --max_epochs 300 \
  --gpu 1 \
  --batch_size 8 \
  --model PraNet \
  --base_dir ./data/ACDC \
  --dataset_name ACDC \
  --num_classes 4 \
  --input_channel 3 \
  --val_interval 100


python main_multi3d.py \
  --max_epochs 300 \
  --gpu 1 \
  --batch_size 8 \
  --model GH_UNet \
  --base_dir ./data/ACDC \
  --dataset_name ACDC \
  --num_classes 4 \
  --input_channel 3 \
  --val_interval 100