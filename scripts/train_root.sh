#!/bin/bash

# 从命令行参数获取模型列表和GPU编号
models_input=$1
gpu_id=$2

# 如果没有提供参数，使用默认值
if [ -z "$models_input" ]; then
    model_list=()
else
    # 将输入的模型列表字符串转换为数组
    IFS=',' read -ra model_list <<< "$models_input"
fi

if [ -z "$gpu_id" ]; then
    gpu_id=7
fi


commands=(
"python main.py         --do_deeps False --max_epochs 200 --gpu $gpu_id --batch_size 8 --model %s --base_dir ./data/busi        --dataset_name busi       --zero_shot_base_dir ./data/bus                 --zero_shot_dataset_name bus"
"python main.py         --do_deeps False --max_epochs 200 --gpu $gpu_id --batch_size 8 --model %s --base_dir ./data/DRIVE       --dataset_name DRIVE      --zero_shot_base_dir ./data/CHASEDB1            --zero_shot_dataset_name CHASEDB1"
"python main.py         --do_deeps False --max_epochs 200 --gpu $gpu_id --batch_size 8 --model %s --base_dir ./data/isic18      --dataset_name isic18     --zero_shot_base_dir ./data/PH2-test            --zero_shot_dataset_name PH2-test"
"python main.py         --do_deeps False --max_epochs 200 --gpu $gpu_id --batch_size 8 --model %s --base_dir ./data/Kvasir-SEG  --dataset_name Kvasir-SEG --zero_shot_base_dir ./data/Kvasir-test         --zero_shot_dataset_name cvc300     --val_file_dir CVC-300val.txt"
"python main.py         --do_deeps False --max_epochs 200 --gpu $gpu_id --batch_size 8 --model %s --base_dir ./data/tuscui      --dataset_name tuscui     --zero_shot_base_dir ./data/TUCC-test/malignant --zero_shot_dataset_name malignant"
"python main.py         --do_deeps False --max_epochs 200 --gpu $gpu_id --batch_size 8 --model %s --base_dir ./data/DSB2018     --dataset_name DSB2018    --zero_shot_base_dir ./data/MonuSeg2018-test    --zero_shot_dataset_name MonuSeg2018-test"
"python main.py         --do_deeps False --max_epochs 200 --gpu $gpu_id --batch_size 8 --model %s --base_dir ./data/Montgomery  --dataset_name Montgomery --zero_shot_base_dir ./data/NIH-test            --zero_shot_dataset_name NIH-test"
"python main_multi3d.py --do_deeps False --max_epochs 200 --gpu $gpu_id --batch_size 8 --model %s --base_dir ./data/ACDC        --dataset_name ACDC       --num_classes 4 --input_channel 1"
"python main_multi3d.py --do_deeps False --max_epochs 200 --gpu $gpu_id --batch_size 8 --model %s --base_dir ./data/synapse     --dataset_name synapse    --num_classes 9 --input_channel 1 --val_interval 10"
)

# 遍历模型列表
for model in "${model_list[@]}"; do
    # 遍历命令模板
    for cmd_template in "${commands[@]}"; do
        # 使用 printf 替换 %s 为当前模型名，修正变量引用
        cmd=$(printf "$cmd_template" "$model")
        # 执行命令
        eval "$cmd"
    done
done
# Glas BUSBRA
# python main_debug.py         --max_epochs 200 --gpu 7 --batch_size 16 --model UNext --base_dir ./data/Glas      --dataset_name Glas   --exp_name tem  
# python main_debug.py         --max_epochs 200 --gpu 7 --batch_size 16 --model UNext --base_dir ./data/BUSBRA      --dataset_name BUSBRA   --exp_name tem  