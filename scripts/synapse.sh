#!/bin/bash

python main_multi3d.py --max_epochs 300 --gpu 0 --batch_size 8 --model UltraLight_VM_UNet   --base_dir ./data/synapse  --dataset_name synapse  --num_classes 9 --input_channel 3 --val_interval 100 &

python main_multi3d.py --max_epochs 300 --gpu 7 --batch_size 8 --model AC_MambaSeg         --base_dir ./data/synapse  --dataset_name synapse  --num_classes 9 --input_channel 3 --val_interval 100 &

python main_multi3d.py --max_epochs 300 --gpu 1 --batch_size 8 --model H_vmunet           --base_dir ./data/synapse  --dataset_name synapse  --num_classes 9 --input_channel 3 --val_interval 100 &

python main_multi3d.py --max_epochs 300 --gpu 2 --batch_size 8 --model MambaUnet          --base_dir ./data/synapse  --dataset_name synapse  --num_classes 9 --input_channel 3 --val_interval 100 &

python main_multi3d.py --max_epochs 300 --gpu 3 --batch_size 8 --model MUCM_Net           --base_dir ./data/synapse  --dataset_name synapse  --num_classes 9 --input_channel 3 --val_interval 100 &

python main_multi3d.py --max_epochs 300 --gpu 4 --batch_size 8 --model Swin_umamba        --base_dir ./data/synapse  --dataset_name synapse  --num_classes 9 --input_channel 3 --val_interval 100 &

python main_multi3d.py --max_epochs 300 --gpu 5 --batch_size 8 --model Swin_umambaD       --base_dir ./data/synapse  --dataset_name synapse  --num_classes 9 --input_channel 3 --val_interval 100 &

python main_multi3d.py --max_epochs 300 --gpu 6 --batch_size 8 --model VMUNet             --base_dir ./data/synapse  --dataset_name synapse  --num_classes 9 --input_channel 3 --val_interval 100 &

python main_multi3d.py --max_epochs 300 --gpu 0 --batch_size 8 --model VMUNetV2           --base_dir ./data/synapse  --dataset_name synapse  --num_classes 9 --input_channel 3 --val_interval 100