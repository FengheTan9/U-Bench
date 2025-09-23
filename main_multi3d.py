# 标准库导入
import os
import random
import argparse
import logging
import sys
import time
import json
import csv
import tempfile
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import wandb
import torch.nn.functional as F

cpu_num = 1
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default="7", help='gpu')
temp_args, _ = parser.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = temp_args.gpu
print(f"Set CUDA_VISIBLE_DEVICES to {os.environ['CUDA_VISIBLE_DEVICES']}")

import torch
torch.set_num_threads(cpu_num)
import torch.serialization
from medpy.metric import dc,hd95
import torch.nn as nn
import torch.optim as optim
from monai.losses import DiceCELoss
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms

device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

from utils.util import one_hot_encoder, DiceLoss, test_single_volume
from models import build_model
from dataloader.dataloader import getDataloader



def convert_to_numpy(data):

    if isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    elif isinstance(data, dict):
        return {key: convert_to_numpy(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_to_numpy(item) for item in data]
    else:
        return data
def seed_torch(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # GPU随机种子确定
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.use_deterministic_algorithms(True,warn_only=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="CMUNeXt", help='model')
    parser.add_argument('--base_dir', type=str, default="./data/ACDC", help='data base dir')
    parser.add_argument('--dataset_name', type=str, default="ACDC", help='dataset_name')
    parser.add_argument('--train_file_dir', type=str, default="train.txt", help='train_file_dir')
    parser.add_argument('--val_file_dir', type=str, default="val.txt", help='val_file_dir')
    parser.add_argument('--base_lr', type=float, default=0.01,
                        help='segmentation network learning rate')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch_size per gpu')
    parser.add_argument('--gpu', type=str, default="7", help='gpu')
    parser.add_argument('--max_iterations', type=int,


                        default=30000, help='maximum epoch number to train')
    parser.add_argument('--max_epochs', type=int, default=150, help='maximum epoch number to train')
    parser.add_argument('--seed', type=int, default=41, help='seed')
    parser.add_argument('--img_size', type=int, default=256, help='img_size')
    parser.add_argument('--num_classes', type=int, default=4, help='img_size')
    parser.add_argument('--input_channel', type=int, default=3, help='img_size')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
    parser.add_argument('--z_spacing', type=float, default=1, help='z_spacing of CT scan')
    parser.add_argument('--do_deeps', type=bool, default=False, help='Use deep supervision')


    parser.add_argument('--val_interval', type=int, default=1, help='val_interval')
    parser.add_argument('--exp_name', type=str, default="default_exp", help='Experiment name')
    parser.add_argument('--zero_shot_base_dir', type=str, default="", help='zero_base_dir')
    parser.add_argument('--zero_shot_dataset_name', type=str, default="", help='zero_shot_dataset_name')
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
    parser.add_argument('--model_id', type=int, default=0, help='model_id')
    parser.add_argument('--just_for_test', type=bool, default=0, help='just for test')

    args = parser.parse_args()

    print(f"Set CUDA_VISIBLE_DEVICES to {os.environ['CUDA_VISIBLE_DEVICES']}")

    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        print(f"Current CUDA device index: {current_device}")
    else:
        print("CUDA is not available.")

    seed_torch(args.seed)
    return args

args = parse_arguments()



def deep_supervision_loss(outputs, label_batch, ce_loss,dice_loss,weights=None):

    num=len(outputs)


    total_loss = 0.0


    for i, output in enumerate(outputs):
        loss_ce = ce_loss(output, label_batch[:].long())
        loss_dice = dice_loss(output, label_batch, softmax=True)

        loss = 0.3 * loss_ce + 0.7 * loss_dice

        total_loss += loss

    return total_loss/ num 

def load_model(args, model_best_or_final="best"):
    exp_save_dir= args.exp_save_dir
    model = build_model(args, input_channel=args.input_channel, num_classes=args.num_classes)

    if model_best_or_final == "best":
        model_path = os.path.join(exp_save_dir, f'checkpoint_best.pth')
    else:
        model_path = os.path.join(exp_save_dir, f'checkpoint_final.pth')

    torch.serialization.add_safe_globals({
        'torch': torch,
        'torch.nn': torch.nn,
        'torch.optim': torch.optim
    })

    checkpoint = torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), weights_only=False)

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    return model, model_path

def inference(args, model, testloader,logger, test_save_path=None):
    logger.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    with torch.no_grad():
        for i_batch, sampled_batch in tqdm(enumerate(testloader)):
            h, w = sampled_batch["image"].size()[2:]
            image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
            metric_i = test_single_volume(image=image, label=label, net=model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                          test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing, do_deeps=args.do_deeps)
            metric_list += np.array(metric_i)
            logger.info('idx %d case %s mean_dice %f mean_hd95 %f, mean_jacard %f mean_asd %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1], np.mean(metric_i, axis=0)[2], np.mean(metric_i, axis=0)[3]))
        metric_list = metric_list / len(testloader)
        for i in range(1, args.num_classes):
            logger.info('Mean class (%d) mean_dice %f mean_hd95 %f, mean_jacard %f mean_asd %f' % (i, metric_list[i-1][0], metric_list[i-1][1], metric_list[i-1][2], metric_list[i-1][3]))
        performance = np.mean(metric_list, axis=0)[0]
        mean_hd95 = np.mean(metric_list, axis=0)[1]
        mean_jacard = np.mean(metric_list, axis=0)[2]
        mean_asd = np.mean(metric_list, axis=0)[3]
        logger.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f, mean_jacard : %f mean_asd : %f' % (performance, mean_hd95, mean_jacard, mean_asd))
        return performance, mean_hd95, mean_jacard, mean_asd
        
def val(valloader,net,Best_dcs ):
    logging.info("Validation ===>")
    dc_sum=0
    metric_list = 0.0
    net.eval()
    for i, val_sampled_batch in enumerate(valloader):
        val_image_batch, val_label_batch = val_sampled_batch["image"], val_sampled_batch["label"]
        val_image_batch, val_label_batch = val_image_batch.type(torch.FloatTensor), val_label_batch.type(torch.FloatTensor)
        val_image_batch, val_label_batch = val_image_batch.cuda().unsqueeze(1), val_label_batch.cuda().unsqueeze(1)
        p1, p2, p3, p4 = net(val_image_batch)
        val_outputs = p1 + p2 + p3 + p4
        val_outputs = torch.argmax(torch.softmax(val_outputs, dim=1), dim=1).squeeze(0)
        
        dc_sum+=dc(val_outputs.cpu().data.numpy(),val_label_batch[:].cpu().data.numpy())
    performance = dc_sum / len(valloader)
    logging.info('Testing performance in val model: mean_dice : %f, best_dice : %f' % (performance, Best_dcs))

    print("val avg_dsc: %f" % (performance))
    return performance 
def init_dir(args):
    exp_save_dir = f'./output/{args.model}/{args.dataset_name}/{args.exp_name}/'
    os.makedirs(exp_save_dir, exist_ok=True)
    args.exp_save_dir = exp_save_dir

    config_file_path = os.path.join(exp_save_dir, f'config.json')
    args_dict = vars(args)
    with open(config_file_path, 'w') as f:
        json.dump(args_dict, f, indent=4)
    print(f"Config saved to {config_file_path}")

    writer = SummaryWriter(log_dir=f'{exp_save_dir}/tensorboard_logs/')
    log_file = os.path.join(exp_save_dir, f'training.log')

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    model = build_model(config=args,input_channel=args.input_channel, num_classes=args.num_classes).to(device)


    return exp_save_dir, writer, logger, model#, wandb

def trainer_multi3d(args,exp_save_dir, writer, logger, model):


    start_epoch = 0

    base_lr = args.base_lr
    trainloader, valloader = getDataloader(args)

    args.batch_size = args.batch_size * args.n_gpu

    model_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"model:{args.model} model_parameters:{model_parameters}")
    logger.info(f"train file dir:{args.train_file_dir} val file dir:{args.val_file_dir}")
    logger.info(f"{len(trainloader)} iterations per epoch")

    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(args.num_classes)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    if args.resume:
        checkpoint_path = os.path.join(exp_save_dir, f'checkpoint.pth')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            logger.info(f"Resuming training from epoch {start_epoch} ")

    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    logger.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0

    train_loss_list = []
    val_performance_list = []

    best_metric={
        "best_dice":0.0,
        "best_dice_with_hd95":0.0,
        "best_dice_with_jacard":0.0,
        "best_dice_with_asd":0.0,
    }
    final_metric={
        "final_dice":0.0,
        "final_hd95":0.0,
        "final_jacard":0.0,
        "final_asd":0.0,
    }
    for epoch_num in tqdm(range(start_epoch, max_epoch), desc='Training Progress'):
        epoch_loss = 0.0
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.to(device), label_batch.to(device)

            output = model(image_batch)
            if args.do_deeps:
                loss = deep_supervision_loss(outputs=output, label_batch=label_batch, ce_loss=ce_loss,dice_loss=dice_loss)
            else:
                loss_ce = ce_loss(output, label_batch[:].long())
                loss_dice = dice_loss(output, label_batch, softmax=True)
                loss = 0.3 * loss_ce + 0.7 * loss_dice # from [URL]

            optimizer.zero_grad()
            loss.backward() 
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(trainloader)
        train_loss_list.append(avg_epoch_loss)
        logger.info('epoch %d : average loss : %f, lr: %f' % (epoch_num, avg_epoch_loss, lr_))

        if (epoch_num + 1) % args.val_interval == 0:
            performance, mean_hd95, mean_jacard, mean_asd = inference(args=args,model=model,logger=logger,testloader=valloader)                                
            final_metric["final_dice"]=performance
            final_metric["final_hd95"]=mean_hd95
            final_metric["final_jacard"]=mean_jacard
            final_metric["final_asd"]=mean_asd
            val_performance_list.append(performance)
            if performance > best_performance:
                best_metric["best_dice"]=performance
                best_metric["best_dice_with_hd95"]=mean_hd95
                best_metric["best_dice_with_jacard"]=mean_jacard
                best_metric["best_dice_with_asd"]=mean_asd
                best_performance = performance
                model_save_path = os.path.join(
                    exp_save_dir,
                    f'checkpoint_best.pth'
                )

                torch.save({
                    'state_dict': model.state_dict(),
                    'config': vars(args),
                    'epoch': epoch_num + 1,
                    'best_performance':best_performance,
                }, model_save_path)
                logger.info("=> saved best model with config")
            model.train()

        if epoch_num == args.max_epochs - 1:
            final_model_save_path = os.path.join(
                exp_save_dir,
                f'checkpoint_final.pth'
            )
            torch.save({
                'state_dict': model.state_dict(),
                'final_performance': final_metric["final_dice"],
                'config': vars(args),
                'epoch': epoch_num + 1,}, final_model_save_path)
            logger.info("=> saved final model with config")

        logger.info(f"current epoch:{epoch_num}")
        logger.info(f"best_dice:{best_metric['best_dice']} best_dice_with_hd95:{best_metric['best_dice_with_hd95']} best_dice_with_jacard:{best_metric['best_dice_with_jacard']} best_dice_with_asd:{best_metric['best_dice_with_asd']} ")
        logger.info(f"final_dice:{final_metric['final_dice']} final_hd95:{final_metric['final_hd95']} final_jacard:{final_metric['final_jacard']} final_asd:{final_metric['final_asd']} ")


        

    writer.close()


    fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    epochs = list(range(len(train_loss_list)))

    axs[0].plot(epochs, train_loss_list)
    axs[0].set_title('Training Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')

    axs[1].plot(epochs[::args.val_interval], val_performance_list)
    axs[1].set_title('Validation Performance')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Performance')
    plt.tight_layout()

    plt.savefig(os.path.join(exp_save_dir, 'training_progress.png'))

    best_metric=convert_to_numpy(best_metric)
    final_metric=convert_to_numpy(final_metric)

    logger.info(f"Training completed. Best Dice: {best_metric['best_dice']}, Best HD95: {best_metric['best_dice_with_hd95']}, Best Jaccard: {best_metric['best_dice_with_jacard']}, Best ASD: {best_metric['best_dice_with_asd']}")
    logger.info(f"Final Dice: {final_metric['final_dice']}, Final HD95: {final_metric['final_hd95']}, Final Jaccard: {final_metric['final_jacard']}, Final ASD: {final_metric['final_asd']}")

    return best_metric,final_metric




if __name__ == "__main__":


    print(f"\n=== Testing model: {args.model} ===")
    exp_save_dir, writer, logger, model = init_dir(args)

    row_data=vars(args)
    if args.just_for_test:
        csv_file = f"./result/result_{args.dataset_name}_test.csv"
        file_exists = os.path.isfile(csv_file)
        val_metric={}
        trainloader, valloader = getDataloader(args)
        performance, mean_hd95, mean_jacard, mean_asd = inference(args=args,model=model,logger=logger,testloader=valloader)                                
        val_metric["final_dice"]=performance
        val_metric["final_hd95"]=mean_hd95
        val_metric["final_jacard"]=mean_jacard
        val_metric["final_asd"]=mean_asd
        if val_metric:
            row_data.update(val_metric)
        with open(csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row_data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row_data)
        exit()
    #try:
    csv_file = f"./result/result_{args.dataset_name}_train.csv"
    file_exists = os.path.isfile(csv_file)
    best_metric,final_metric = trainer_multi3d(args,exp_save_dir, writer, logger, model)        
    print("Best performance: ", best_metric)
    row_data.update(best_metric)
    row_data.update(final_metric)
   
    with open(csv_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)

    
    print(f"Model {args.model} tested successfully")