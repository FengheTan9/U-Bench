#!/bin/bash


MODEL=U_Net

python inference_case.py --max_epochs 300 --gpu 0 --batch_size 8 --just_for_test True --model $MODEL --base_dir ./data/busi --dataset_name busi --zero_shot_base_dir ./data/bus --zero_shot_dataset_name bus

python inference_case.py --max_epochs 300 --gpu 0 --batch_size 8 --just_for_test True --model $MODEL --base_dir ./data/BUSBRA --dataset_name BUSBRA --zero_shot_base_dir ./data/bus --zero_shot_dataset_name bus

python inference_case.py --max_epochs 300 --gpu 0 --batch_size 8 --just_for_test True --model $MODEL --base_dir ./data/tuscui --dataset_name tuscui --zero_shot_base_dir ./data/TUCC-test/malignant --zero_shot_dataset_name malignant

python inference_case.py --max_epochs 300 --gpu 0 --batch_size 8 --just_for_test True --model $MODEL --base_dir ./data/isic18 --dataset_name isic18 --zero_shot_base_dir ./data/PH2-test --zero_shot_dataset_name PH2-test

python inference_case.py --max_epochs 300 --gpu 0 --batch_size 8 --just_for_test True --model $MODEL --base_dir ./data/uwaterlooskincancer --dataset_name uwaterlooskincancer

python inference_case.py --max_epochs 300 --gpu 0 --batch_size 8 --just_for_test True --model $MODEL --base_dir ./data/Kvasir-SEG --dataset_name Kvasir-SEG --zero_shot_base_dir ./data/Kvasir-test --zero_shot_dataset_name cvc300 --val_file_dir CVC-300val.txt

python inference_case.py --max_epochs 300 --gpu 0 --batch_size 8 --just_for_test True --model $MODEL --dataset_name CHASEDB1 --base_dir ./data/CHASEDB1 --zero_shot_dataset_name DRIVE --zero_shot_base_dir ./data/DRIVE

python inference_case.py --max_epochs 300 --gpu 0 --batch_size 8 --just_for_test True --model $MODEL --base_dir ./data/DRIVE --dataset_name DRIVE --zero_shot_base_dir ./data/CHASEDB1 --zero_shot_dataset_name CHASEDB1

python inference_case.py --max_epochs 300 --gpu 0 --batch_size 8 --just_for_test True --model $MODEL --base_dir ./data/DSB2018 --dataset_name DSB2018 --zero_shot_base_dir ./data/MonuSeg2018-test --zero_shot_dataset_name MonuSeg2018-test

python inference_case.py --max_epochs 300 --gpu 0 --batch_size 8 --just_for_test True --model $MODEL --base_dir ./data/Glas --dataset_name Glas

python inference_case.py --max_epochs 300 --gpu 0 --batch_size 8 --just_for_test True --model $MODEL --base_dir ./data/monusac --dataset_name monusac

python inference_case.py --max_epochs 300 --gpu 0 --batch_size 8 --just_for_test True --model $MODEL --base_dir ./data/cellnuclei --dataset_name cellnuclei

python inference_case.py --max_epochs 300 --gpu 0 --batch_size 8 --just_for_test True --model $MODEL --base_dir ./data/covidquex --dataset_name covidquex --zero_shot_base_dir ./data/mosmedplus  --zero_shot_dataset_name mosmedplus

python inference_case.py --max_epochs 300 --gpu 0 --batch_size 8 --just_for_test True --model $MODEL --base_dir ./data/Montgomery --dataset_name Montgomery --zero_shot_base_dir ./data/NIH-test --zero_shot_dataset_name NIH-test

python inference_case.py --max_epochs 300 --gpu 0 --batch_size 8 --just_for_test True --model $MODEL --base_dir ./data/dca1 --dataset_name dca1

python inference_case.py --max_epochs 300 --gpu 0 --batch_size 8 --just_for_test True --model $MODEL --base_dir ./data/cystoidfluid --dataset_name cystoidfluid