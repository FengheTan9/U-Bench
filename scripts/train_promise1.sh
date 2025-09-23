#!/bin/bash

models_input=(RWKV_UNet U_RWKV DS_TransUNet Zig_RiR)
gpu_id=$1

if [ -z "$gpu_id" ]; then
    gpu_id=7
fi

commands=(
"python main.py --max_epochs 300 --gpu $gpu_id --batch_size 8 --model %s --base_dir ./data/promise12 --dataset_name promise12"
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