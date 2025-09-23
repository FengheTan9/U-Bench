import os

cpu_num = 1
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)



import random
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import logging
import numpy as np
import torch

torch.set_num_threads(cpu_num)
torch.multiprocessing.set_sharing_strategy('file_system')

import wandb
import torch.optim as optim
from monai.losses import DiceCELoss, DiceLoss
import csv
device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
from models import build_model
import utils.losses as losses
from utils.metrics_medpy import get_metrics
from utils.util import AverageMeter
import tempfile
from torch.utils.tensorboard import SummaryWriter
from dataloader.dataloader import getDataloader,getZeroShotDataloader
import torch.nn.functional as F


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default="7", help='gpu')
temp_args, _ = parser.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = temp_args.gpu
print(f"Set CUDA_VISIBLE_DEVICES to {os.environ['CUDA_VISIBLE_DEVICES']}")



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
    torch.cuda.manual_seed(seed)
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
    parser.add_argument('--model', type=str, default="U_Net", help='model')
    parser.add_argument('--base_dir', type=str, default="./data/busi", help='data base dir')
    parser.add_argument('--dataset_name', type=str, default="busi", help='dataset_name')
    parser.add_argument('--train_file_dir', type=str, default="train.txt", help='train_file_dir')
    parser.add_argument('--val_file_dir', type=str, default="val.txt", help='val_file_dir')
    parser.add_argument('--base_lr', type=float, default=0.01,
                        help='segmentation network learning rate')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch_size per gpu')
    parser.add_argument('--gpu', type=str, default="7", help='gpu')
    parser.add_argument('--max_epochs', type=int, default=2, help='epoch')
    parser.add_argument('--seed', type=int, default=41, help='seed')
    parser.add_argument('--img_size', type=int, default=256, help='img_size')
    parser.add_argument('--num_classes', type=int, default=1, help='img_size')
    parser.add_argument('--input_channel', type=int, default=3, help='img_size')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--exp_name', type=str, default="default_exp", help='Experiment name')
    parser.add_argument('--zero_shot_base_dir', type=str, default="", help='zero_base_dir')
    parser.add_argument('--zero_shot_dataset_name', type=str, default="", help='zero_shot_dataset_name')
    parser.add_argument('--do_deeps', type=bool, default=False, help='Use deep supervision')
    parser.add_argument('--model_id', type=int, default=0, help='model_id')
    parser.add_argument('--just_for_test', type=bool, default=0, help='just for test')
    parser.add_argument('--just_for_zero_shot', type=bool, default=0, help='just for test')
    args = parser.parse_args()
    seed_torch(args.seed)
    return args


args = parse_arguments()



def deep_supervision_loss(outputs, label_batch, loss_metric,weights=None):
    num=len(outputs)

    total_loss = 0.0

    for i, output in enumerate(outputs):
        if output.shape[1:] != label_batch.shape[1:]:
            output = F.interpolate(output, size=label_batch.shape[1:], mode='bilinear', align_corners=True)
        loss = loss_metric(output, label_batch)
        total_loss += loss

    return total_loss/ num

def load_model(args, model_best_or_final="best"):
    exp_save_dir= args.exp_save_dir
    model = build_model(args, input_channel=args.input_channel, num_classes=args.num_classes).to(device)
    if model_best_or_final == "best":
        model_path = os.path.join(exp_save_dir, f'checkpoint_best.pth')

    else:
        model_path = os.path.join(exp_save_dir, f'checkpoint_final.pth')

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)

    model.to(device)

    return model, model_path

def zero_shot(args,logger,model=None,wandb=None):
    valloader = getZeroShotDataloader(args)
    if model is None:
        model,model_path = load_model(args)

    logger.info("train file dir:{} val file dir:{}".format(args.train_file_dir, args.val_file_dir))
    criterion = losses.__dict__['BCEDiceLoss']().to(device)

    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'val_loss': AverageMeter(),
                  'val_iou': AverageMeter(),
                  'SE': AverageMeter(),
                  'PC': AverageMeter(),
                  'F1': AverageMeter(),
                  'ACC': AverageMeter()
                  }
    model.eval()

    with torch.no_grad():
        for i_batch, sampled_batch in tqdm(enumerate(valloader), total=len(valloader), desc="Zero-shot Validation"):
            input, target = sampled_batch['image'], sampled_batch['label']
            input = input.to(device)
            target = target.to(device)
            output = model(input)
            output = output[-1] if args.do_deeps else output
            loss = criterion(output, target)
            
            iou, _, SE, PC, F1, _, ACC = get_metrics(output, target)
            avg_meters['val_loss'].update(loss.item(), input.size(0))
            avg_meters['val_iou'].update(iou, input.size(0))
            avg_meters['SE'].update(SE, input.size(0))
            avg_meters['PC'].update(PC, input.size(0))
            avg_meters['F1'].update(F1, input.size(0))
            avg_meters['ACC'].update(ACC, input.size(0))
    logger.info(f"zero shot on {args.zero_shot_dataset_name}")
    logger.info('val_loss %.4f - val_iou %.4f - val_SE %.4f - val_PC %.4f - val_F1 %.4f - val_ACC %.4f'
        % (avg_meters['val_loss'].avg, avg_meters['val_iou'].avg, avg_meters['SE'].avg,
            avg_meters['PC'].avg, avg_meters['F1'].avg, avg_meters['ACC'].avg))

    
    zero_shot_result = {"zeroshot_loss":avg_meters['val_loss'].avg, "zeroshot_iou":avg_meters['val_iou'].avg, "zeroshot_SE":avg_meters['SE'].avg,
            "zeroshot_PC":avg_meters['PC'].avg, "zeroshot_F1":avg_meters['F1'].avg, "zeroshot_ACC":avg_meters['ACC'].avg}
    zero_shot_result = convert_to_numpy(zero_shot_result)
    return zero_shot_result


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


def validate(args,logger,model):
    trainloader,valloader = getDataloader(args)
    criterion = losses.__dict__['BCEDiceLoss']().to(device)
    avg_meters = {'loss': AverageMeter(),
                'iou': AverageMeter(),
                'val_loss': AverageMeter(),
                'val_iou': AverageMeter(),
                'SE': AverageMeter(),
                'PC': AverageMeter(),
                'F1': AverageMeter(),
                'ACC': AverageMeter()
                }
    model.eval()
    with torch.no_grad():
        for i_batch, sampled_batch in enumerate(valloader):
            input, target = sampled_batch['image'], sampled_batch['label']
            input = input.to(device)
            target = target.to(device)
            output = model(input)
            output = output[-1] if args.do_deeps else output
            loss = criterion(output, target)
            
            iou, _, SE, PC, F1, _, ACC = get_metrics(output, target)
            avg_meters['val_loss'].update(loss.item(), input.size(0))
            avg_meters['val_iou'].update(iou, input.size(0))
            avg_meters['SE'].update(SE, input.size(0))
            avg_meters['PC'].update(PC, input.size(0))
            avg_meters['F1'].update(F1, input.size(0))
            avg_meters['ACC'].update(ACC, input.size(0))

    val_metric_dict = {
        "val_loss":avg_meters['val_loss'].avg, "val_iou":avg_meters['val_iou'].avg, "val_SE":avg_meters['SE'].avg,
            "val_PC":avg_meters['PC'].avg, "val_F1":avg_meters['F1'].avg, "val_ACC":avg_meters['ACC'].avg
    }
    val_metric_dict = convert_to_numpy(val_metric_dict)
    return val_metric_dict



def train(args,exp_save_dir, writer, logger, model):
    start_epoch = 0
    base_lr = args.base_lr
    trainloader, valloader = getDataloader(args)

    model_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"model:{args.model} model_parameters:{model_parameters}")
    logger.info(f"train file dir:{args.train_file_dir} val file dir:{args.val_file_dir}")
    logger.info(f"{len(trainloader)} iterations per epoch")
    
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    criterion = losses.__dict__['BCEDiceLoss']().to(device)


    train_metric_dict = {
            "best_iou": 0,
            "best_epoch": 0,
            "best_iou_withSE": 0,
            "best_iou_withPC": 0,
            "best_iou_withF1": 0,
            "best_iou_withACC": 0,
            "last_iou": 0,
            "last_SE": 0,
            "last_PC": 0,
            "last_F1": 0,
            "last_ACC": 0
    }

    max_epoch = args.max_epochs
    max_iterations = len(trainloader) * max_epoch

    train_loss_list = []
    train_iou_list = []
    loss_list = []
    iou_list = []
    f1_list = []

    if args.resume:
        checkpoint_path = os.path.join(exp_save_dir, f'checkpoint.pth')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            train_metric_dict["best_iou"] = checkpoint['best_iou']
            logger.info(f"Resuming training from epoch {start_epoch} with best IoU {train_metric_dict['best_iou']}")

    iter_num = start_epoch * len(trainloader)

    for epoch_num in tqdm(range(start_epoch, max_epoch), desc='Training Progress'):
        model.train()
        avg_meters = {'loss': AverageMeter(),
                      'iou': AverageMeter(),
                      'val_loss': AverageMeter(),
                      'val_iou': AverageMeter(),
                      'SE': AverageMeter(),
                      'PC': AverageMeter(),
                      'F1': AverageMeter(),
                      'ACC': AverageMeter()
                      }

        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.to(device), label_batch.to(device)

            if args.do_deeps:
                outputs = model(volume_batch)
                loss = deep_supervision_loss(outputs=outputs,label_batch=label_batch,loss_metric=criterion)
                outputs=outputs[-1]
            else:
                outputs = model(volume_batch)
                loss = criterion(outputs, label_batch)


            iou, dice, _, _, _, _, _ = get_metrics(outputs, label_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num += 1
            avg_meters['loss'].update(loss.item(), volume_batch.size(0))
            avg_meters['iou'].update(iou, volume_batch.size(0))

        model.eval()
        with torch.no_grad():
            for i_batch, sampled_batch in enumerate(valloader):
                input, target = sampled_batch['image'], sampled_batch['label']
                input = input.to(device)
                target = target.to(device)
                output = model(input)
                output = output[-1] if args.do_deeps else output
                loss = criterion(output, target)
                
                iou, _, SE, PC, F1, _, ACC = get_metrics(output, target)
                avg_meters['val_loss'].update(loss.item(), input.size(0))
                avg_meters['val_iou'].update(iou, input.size(0))
                avg_meters['SE'].update(SE, input.size(0))
                avg_meters['PC'].update(PC, input.size(0))
                avg_meters['F1'].update(F1, input.size(0))
                avg_meters['ACC'].update(ACC, input.size(0))

        train_loss_list.append(avg_meters['loss'].avg)
        loss_list.append(avg_meters['val_loss'].avg)
        if isinstance(avg_meters['val_iou'].avg, torch.Tensor):
            iou_list.append(avg_meters['val_iou'].avg.cpu().numpy())
        else:
            iou_list.append(avg_meters['val_iou'].avg)
        if isinstance(avg_meters['F1'].avg, torch.Tensor):
            f1_list.append(avg_meters['F1'].avg.cpu().numpy())
        else:
            f1_list.append(avg_meters['F1'].avg)

        writer.add_scalar('Train/Loss', avg_meters['loss'].avg, epoch_num)
        writer.add_scalar('Train/IOU', avg_meters['iou'].avg, epoch_num)
        writer.add_scalar('Val/Loss', avg_meters['val_loss'].avg, epoch_num)
        writer.add_scalar('Val/IOU', avg_meters['val_iou'].avg, epoch_num)
        writer.add_scalar('Val/SE', avg_meters['SE'].avg, epoch_num)
        writer.add_scalar('Val/PC', avg_meters['PC'].avg, epoch_num)
        writer.add_scalar('Val/F1', avg_meters['F1'].avg, epoch_num)
        writer.add_scalar('Val/ACC', avg_meters['ACC'].avg, epoch_num)


        log_info = (
            f"epoch [{epoch_num}/{max_epoch}]  train_loss: {avg_meters['loss'].avg:.4f}, train_iou: {avg_meters['iou'].avg:.4f} "
            f"- val_loss {avg_meters['val_loss'].avg:.4f} - val_iou {avg_meters['val_iou'].avg:.4f} "
            f"- val_SE {avg_meters['SE'].avg:.4f} - val_PC {avg_meters['PC'].avg:.4f} "
            f"- val_F1 {avg_meters['F1'].avg:.4f} - val_ACC {avg_meters['ACC'].avg:.4f}"
        )
        logger.info(log_info)

        if avg_meters['val_iou'].avg > train_metric_dict["best_iou"]:
            train_metric_dict["best_iou"] = avg_meters['val_iou'].avg
            train_metric_dict["best_epoch"] = epoch_num
            train_metric_dict["best_iou_withSE"] = avg_meters['SE'].avg
            train_metric_dict["best_iou_withPC"] = avg_meters['PC'].avg
            train_metric_dict["best_iou_withF1"] = avg_meters['F1'].avg
            train_metric_dict["best_iou_withACC"] = avg_meters['ACC'].avg

            model_save_path = os.path.join(
                exp_save_dir,
                f'checkpoint_best.pth'
            )

            torch.save({
                'state_dict': model.state_dict(),
                'config': vars(args),
                'epoch': epoch_num + 1,
            }, model_save_path)

            print("=> saved best model with config")

        if epoch_num == max_epoch - 1:
            train_metric_dict["last_iou"] = avg_meters['val_iou'].avg
            train_metric_dict["last_SE"] = avg_meters['SE'].avg
            train_metric_dict["last_PC"] = avg_meters['PC'].avg
            train_metric_dict["last_F1"] = avg_meters['F1'].avg
            train_metric_dict["last_ACC"] = avg_meters['ACC'].avg

        checkpoint_path = os.path.join(exp_save_dir, f'checkpoint_final.pth')

        torch.save({
            'epoch': epoch_num + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_iou': train_metric_dict["best_iou"],
            'config': vars(args),
        }, checkpoint_path)


    writer.close()

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    epochs = list(range(len(train_loss_list)))
    # Plot training loss
    axs[0, 0].plot(train_loss_list)
    axs[0, 0].set_title('Training Loss')
    axs[0, 0].set_xlabel('Epoch')
    # Plot validation loss
    axs[0, 1].plot(loss_list)
    axs[0, 1].set_title('Validation Loss')
    axs[0, 1].set_xlabel('Epoch')
    # Plot validation IoU
    axs[1, 0].plot(iou_list)
    axs[1, 0].set_title('Validation IoU')
    axs[1, 0].set_xlabel('Epoch')
    # Plot validation F1
    axs[1, 1].plot(f1_list)
    axs[1, 1].set_title('Validation F1')
    axs[1, 1].set_xlabel('Epoch')
    plt.tight_layout()
    # Save the figure
    plt.savefig(f'{exp_save_dir}/{args.model}_{args.train_file_dir}_{args.batch_size}_{args.max_epochs}_{args.seed}_{args.base_lr}_{train_metric_dict["best_iou"]:.4f}.png')


    train_metric_dict=convert_to_numpy(train_metric_dict)
    logger.info(f"Training completed. Best IoU: {train_metric_dict['best_iou']}, Best Epoch: {train_metric_dict['best_epoch']}, Best SE: {train_metric_dict['best_iou_withSE']}, Best PC: {train_metric_dict['best_iou_withPC']}, Best F1: {train_metric_dict['best_iou_withF1']}, Best ACC: {train_metric_dict['best_iou_withACC']}")
    logger.info(f"Last IoU: {train_metric_dict['last_iou']}, Last SE: {train_metric_dict['last_SE']}, Last PC: {train_metric_dict['last_PC']}, Last F1: {train_metric_dict['last_F1']}, Last ACC: {train_metric_dict['last_ACC']}")

    return train_metric_dict






if __name__ == "__main__":

    
    print(f"\n=== Testing model: {args.model} ===")

    exp_save_dir, writer, logger, model = init_dir(args)
    row_data=vars(args)


    if args.just_for_test:
        if args.zero_shot_dataset_name != "":
            csv_file = f"./result/result_{args.dataset_name}_2_{args.zero_shot_dataset_name}_test.csv"
        else:
            csv_file = f"./result/result_{args.dataset_name}_test.csv"
        
        file_exists = os.path.isfile(csv_file)
        model, model_path = load_model(args, model_best_or_final="best")
        print(f"Just for test, skipping training. loading model form best checkpoint. Model loaded from {model_path}")
        val_metric_dict = validate(args,logger, model)
        if args.zero_shot_dataset_name !="":
            zeroshot_result=zero_shot(args,logger, model)
        else:
            zeroshot_result=None
        if val_metric_dict:
            row_data.update(val_metric_dict)
        if zeroshot_result:
            row_data.update(zeroshot_result)

        with open(csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row_data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row_data)
        exit()
    try:
        csv_file = f"./result/result_{args.dataset_name}_train.csv"
        file_exists = os.path.isfile(csv_file)
        train_metric_dict = train(args,exp_save_dir, writer, logger, model)
        if args.zero_shot_dataset_name !="":
            zeroshot_result=zero_shot(args,logger, model)
        else:
            zeroshot_result=None
        if train_metric_dict:
            row_data.update(train_metric_dict)
        if zeroshot_result:
            row_data.update(zeroshot_result)
        with open(csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row_data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row_data)
        print(f"Model {args.model} tested successfully")
    except Exception as e:
        row_data.update({"Error": str(e)})
        error_row = row_data.copy()
        with open('./ERROR.log', 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=error_row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(error_row)
        print(f"Model {args.model} failed: {str(e)}")

    print(f"Model {args.model} tested successfully")
    