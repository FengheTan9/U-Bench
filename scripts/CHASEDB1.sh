#!/bin/bash

models_input=(Swin_umamba DA_TransUNet MedVKAN AC_MambaSeg MUCM_Net VMUNet)
gpu_id=$1

if [ -z "$gpu_id" ]; then
    gpu_id=7
fi

commands=(
"python main.py --max_epochs 300 --gpu 2 --batch_size 8 --model %s --dataset_name CHASEDB1 --base_dir ./data/CHASEDB1 --zero_shot_dataset_name DRIVE --zero_shot_base_dir ./data/DRIVE"
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