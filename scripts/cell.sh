#!/bin/bash

models_input=(DA_TransUNet FAT_Net HiFormer SCUNet_plus_plus MissFormer BEFUnet H_vmunet)
gpu_id=$1

if [ -z "$gpu_id" ]; then
    gpu_id=7
fi

commands=(
"python main.py --max_epochs 300 --gpu 5 --batch_size 8 --model %s --base_dir ./data/cellnuclei --dataset_name cellnuclei"
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