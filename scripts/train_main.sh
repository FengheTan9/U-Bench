#!/bin/bash
# 运行如下命令


# bash train_root.sh "U_Net,UNext,AttU_Net,CMUNet,CMUNeXt,UNet3plus,UCTransNet,TinyUNet,MISSFormer,TransUnet,MedT,ResNet34UnetPlus,SwinUnet,Utr" 5

bash train_root.sh "CMUNeXt" 5  # yes 

bash train_root.sh "TinyUNet" 5 # yes

bash train_root.sh "UNext,AttU_Net,CMUNet,UNet3plus,UCTransNet" 4 # yes
bash train_root.sh "MISSFormer,TransUnet,MedT" 4 # yes
bash train_root.sh "ResNet34UnetPlus,SwinUnet,Utr" 4

bash train_root.sh "MISSFormer" 4 # yes