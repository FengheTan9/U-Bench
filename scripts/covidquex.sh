#!/bin/bash

models_input=(ESKNet UNet3plus RWKV_UNet U_RWKV DA_TransUNet TransNorm MedT HiFormer LV_UNet Zig_RiR SCUNet_plus_plus MissFormer BEFUnet H_vmunet)
gpu_id=$1

if [ -z "$gpu_id" ]; then
    gpu_id=7
fi

commands=(
"python main.py --max_epochs 300 --gpu 3 --batch_size 8 --model %s --base_dir ./data/covidquex --dataset_name covidquex --zero_shot_base_dir ./data/mosmedplus  --zero_shot_dataset_name mosmedplus"
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