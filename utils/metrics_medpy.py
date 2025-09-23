import os 

import torch
import numpy as np
from medpy.metric.binary import hd, assd, dc, precision, recall, specificity

def get_metrics(output, target):

    output = torch.sigmoid(output).cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    

    output = (output > 0.5).astype(np.uint8)
    target = target.astype(np.uint8)


    dice = dc(output, target)

    intersection = np.sum(output * target)
    union = np.sum(output) + np.sum(target) - intersection
    iou = intersection / union if union > 0 else 0

    SE = recall(output, target)

    PC = precision(output, target)

    SP = specificity(output, target)

    ACC = get_accuracy(output, target)

    F1 = 2 * (PC * SE) / (PC + SE) if (PC + SE) > 0 else 0

    return iou, dice, SE, PC, F1, SP, ACC

def dice_coef(output, target):

    output = torch.sigmoid(output).cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    output = (output > 0.5).astype(np.uint8)
    target = target.astype(np.uint8)
    dice = dc(output, target)
    return dice

def get_accuracy(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT == np.max(GT)  
    corr = np.sum(SR == GT)  
    acc = float(corr) / float(SR.size) 
    return acc
