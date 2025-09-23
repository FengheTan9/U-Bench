#!/bin/bash

models_input=(CFFormer MedVKAN ResU_KAN CFM_UNet MCA_UNet)
gpu_id=$1

if [ -z "$gpu_id" ]; then
    gpu_id=7
fi

commands=(
"python main.py --max_epochs 300 --gpu $gpu_id --batch_size 8 --model %s --base_dir ./data/Kvasir-SEG --dataset_name Kvasir-SEG --zero_shot_base_dir ./data/Kvasir-test --zero_shot_dataset_name cvc300 --val_file_dir CVC-300val.txt --just_for_test True"
)

# 遍历模型列表
for model in "${models_input[@]}"; do
    # 遍历命令模板
    for cmd_template in "${commands[@]}"; do
        # 使用 printf 替换 %s 为当前模型名
        cmd=$(printf "$cmd_template" "$model")
        echo "==> Running: $cmd"
        eval "$cmd"
    done
done