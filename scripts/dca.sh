#!/bin/bash

models_input=(RWKV_UNet U_RWKV DS_TransUNet MSLAU_Net G_CASCADE Zig_RiR BEFUnet H_vmunet Polyp_PVT)
gpu_id=$1

if [ -z "$gpu_id" ]; then
    gpu_id=7
fi

commands=(
"python main.py --max_epochs 300 --gpu 2 --batch_size 8 --model %s --base_dir ./data/dca1 --dataset_name dca1"
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