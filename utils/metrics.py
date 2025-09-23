import os 
import torch
from torchmetrics.classification import BinaryJaccardIndex, BinaryRecall, BinaryPrecision, BinarySpecificity, BinaryAccuracy, BinaryF1Score


def get_metrics(output, target):
    device = output.device
    output = torch.sigmoid(output)
    iou_metric = BinaryJaccardIndex(threshold=0.5).to(device)
    se_metric = BinaryRecall(threshold=0.5).to(device)
    pc_metric = BinaryPrecision(threshold=0.5).to(device)
    sp_metric = BinarySpecificity(threshold=0.5).to(device)
    acc_metric = BinaryAccuracy(threshold=0.5).to(device)
    f1_metric = BinaryF1Score(threshold=0.5).to(device)

    iou = iou_metric(output, target)
    dice = 2 * iou / (1+iou)
    SE = se_metric(output, target)
    PC = pc_metric(output, target)
    SP = sp_metric(output, target)
    ACC = acc_metric(output, target)
    F1 = f1_metric(output, target)

    return iou, dice, SE, PC, F1, SP, ACC

def dice_coef(output, target):
    dice_metric = BinaryDiceCoefficient(threshold=0.5)
    output = torch.sigmoid(output)
    dice = dice_metric(output, target)
    return dice