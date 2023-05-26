import os
import os.path as osp
import time
import math
import json
import argparse
from datetime import timedelta
from argparse import ArgumentParser

import torch
import numpy as np
import random
import wandb
from torch import cuda
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm

from east_dataset import EASTDataset
from dataset import SceneTextDataset
from model import EAST
from early_stopping import EarlyStopping

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def log_wandb(step, loss, metrics):
    wandb.log({"step": step, "loss": loss, "metrics": metrics})

def parse_args():
    parser = ArgumentParser()
    
    parser.add_argument("--config", type=str, default="./config.json", help="config file directory address")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = json.load(f)
    
    # Conventional args
    parser.add_argument('--wandb_name', type=str, default=config["wandb_name"])
    parser.add_argument('--fold_num', type=str, default=config["fold_num"])
    parser.add_argument('--max_epoch', type=int, default=config["max_epoch"])
    parser.add_argument('--learning_rate', type=float, default=config["learning_rate"])
    parser.add_argument('--patience', type=int, default=config["patience"])
    parser.add_argument('--delta', type=float, default=config["delta"])
    parser.add_argument('--save_interval', type=int, default=config["save_interval"])
    parser.add_argument('--seed', type=int, default=config["seed"])
    parser.add_argument('--batch_size', type=int, default=config["batch_size"])
    parser.add_argument('--num_workers', type=int, default=config["num_workers"])
    parser.add_argument('--image_size', type=int, default=config["image_size"])
    parser.add_argument('--input_size', type=int, default=config["input_size"])
    parser.add_argument('--ignore_tags', type=list, default=config["ignore_tags"])
    parser.add_argument('--data_dir', type=str, default=config["data_dir"])
    
    parser.add_argument('--train_ignore_under_threshold', type=int, default=config["train_ignore_under_threshold"])
    parser.add_argument('--train_drop_under_threshold', type=int, default=config["train_drop_under_threshold"])
    parser.add_argument('--train_color_jitter', type=bool, default=config["train_color_jitter"])
    parser.add_argument('--train_normalize', type=bool, default=config["train_normalize"])
    parser.add_argument('--val_ignore_under_threshold', type=int, default=config["val_ignore_under_threshold"])
    parser.add_argument('--val_drop_under_threshold', type=int, default=config["val_drop_under_threshold"])
    parser.add_argument('--val_color_jitter', type=bool, default=config["val_color_jitter"])
    parser.add_argument('--val_normalize', type=bool, default=config["val_normalize"])
    
    parser.add_argument('--resume', type=bool, default=config["resume"])
    parser.add_argument('--weight_name', type=str, default=config["weight_name"])
    parser.add_argument('--run_id', type=str, default=config["run_id"])
    
    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    args = parser.parse_args()
    print(args)
    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')
    return args


def do_training(data_dir, device, image_size, input_size, num_workers, batch_size,
                learning_rate, max_epoch, save_interval, ignore_tags, seed, wandb_name, fold_num,
                patience, delta, train_ignore_under_threshold, train_drop_under_threshold, train_color_jitter,
                train_normalize, val_ignore_under_threshold, val_drop_under_threshold, val_color_jitter,
                val_normalize, resume, weight_name, run_id, config):

    seed_everything(seed)
    
    model_dir = os.path.join('./trained_models',wandb_name)
    data_dir = os.environ.get('SM_CHANNEL_TRAIN', data_dir)


    train_dataset = SceneTextDataset(
        data_dir,
        split=f"train{fold_num}",
        image_size=image_size,
        crop_size=input_size,
        ignore_tags=ignore_tags,
        ignore_under_threshold=train_ignore_under_threshold,
        drop_under_threshold=train_drop_under_threshold,
        color_jitter=train_color_jitter,
        normalize=train_normalize
    )
    
    val_dataset = SceneTextDataset(
        data_dir,
        split=f"val{fold_num}",
        image_size=image_size,
        crop_size=input_size,
        ignore_tags=ignore_tags,
        ignore_under_threshold=val_ignore_under_threshold,
        drop_under_threshold=val_drop_under_threshold,
        color_jitter=val_color_jitter,
        normalize=val_normalize
    )

    train_dataset = EASTDataset(train_dataset)
    val_dataset = EASTDataset(val_dataset)
    num_batches = math.ceil(len(train_dataset) / batch_size) 

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False
    )
    
    if resume:
        dict_run_id = {"run_id": run_id}
        json_data = json.dumps(dict_run_id)
        with open('./wandb/wandb-resume.json', 'w') as f:
            f.write(json_data)
        
        run = wandb.init(entity="oif", project='Data_Centric', name=wandb_name, resume=True)
    else:
        print("Training Starting...")
        run = wandb.init(entity="oif", project='Data_Centric', name=wandb_name, resume=None)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch // 2], gamma=0.1)
    early_stopping = EarlyStopping(patience=patience, delta=delta, mode='min', verbose=False)

    prev_ckpt_fpath = osp.join(model_dir, f'best_0epoch.pth') # dummy
    
    best_val_loss = np.inf
    starting_epoch = 0
    
    if wandb.run.resumed:
        checkpoint = torch.load(os.path.join(model_dir, weight_name))   
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        starting_epoch = checkpoint['epoch']
        try:
            best_val_loss = checkpoint['best_validation_loss']
        except:
            pass
        print(f"Resuming Training From Epoch {starting_epoch}...")
    
    for epoch in range(max_epoch):
        train_start = time.time()
        model.train()
        epoch_loss, epoch_loss_cls, epoch_loss_angle, epoch_loss_iou = 0, 0, 0, 0
        train_loss = 0
        val_loss = 0
        print()
        
        with tqdm(total=num_batches) as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
                pbar.set_description('[Epoch {}]'.format(epoch + 1)) # 진행 바 왼쪽에 Epoch 진행 상황 추가

                loss, extra_info1 = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                epoch_loss += train_loss
                epoch_loss_cls += extra_info1['cls_loss']
                epoch_loss_angle += extra_info1['angle_loss']
                epoch_loss_iou += extra_info1['iou_loss']

                pbar.update(1) # 수동으로 진행률을 1씩 증가 시킨다
                train_dict = {
                    '(Train)Class loss': extra_info1['cls_loss'], '(Train)Angle loss': extra_info1['angle_loss'],
                    '(Train)IoU loss': extra_info1['iou_loss']
                }
                pbar.set_postfix(train_dict) # 진행 바 오른쪽에 Train Loss 설명 추가

                mean_train_loss = train_loss / num_batches
                 
                # wandb 로그 기록
                step = epoch * num_batches + pbar.n
                log_wandb(step, train_loss, train_dict)
                
                train_loss = 0
                
            with torch.no_grad():
                val_start = time.time()
                model.eval()      
                print("\nEvaluating validation results...")
                for img, gt_score_map, gt_geo_map, roi_mask in val_loader:
                    loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                    temp = loss.item()
                    val_loss += temp
                    wandb.log({"val_loss": temp})

                mean_val_loss = val_loss / num_batches
                if best_val_loss > mean_val_loss:
                    best_val_loss = mean_val_loss
                    best_val_loss_epoch = epoch+1
                    if not osp.exists(model_dir):
                        os.makedirs(model_dir)
                    
                    ckpt_fpath = osp.join(model_dir, f'best_{epoch+1}epoch.pth')
                    torch.save({
                        'epoch': epoch+1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_validation_loss': best_val_loss,
                        'loss': loss,
                        'step': step,
                        }, ckpt_fpath)
                    if osp.exists(prev_ckpt_fpath):
                        os.remove(prev_ckpt_fpath)
                    prev_ckpt_fpath = osp.join(model_dir, f'best_{epoch+1}epoch.pth')

                print(f'(Val) Class loss: {extra_info["cls_loss"]:.4f}, (Val) Angle loss: {extra_info["angle_loss"]:.4f}, (Val) IoU loss: {extra_info["iou_loss"]:.4f}')

            print('(Train) Mean loss: {:.4f} | Elapsed time: {}'.format(mean_train_loss, timedelta(seconds=time.time() - train_start)))
            print('(Val)   Mean loss: {:.4f} | Elapsed time: {}'.format(mean_val_loss, timedelta(seconds=time.time() - val_start)))
            print('Best Validation Loss: {:.4f} at Epoch {}'.format(best_val_loss, best_val_loss_epoch))

            early_stopping(mean_val_loss)
            if early_stopping.early_stop:
                raise SystemExit(0)

        scheduler.step()
        
        mean_loss = epoch_loss / num_batches
        mean_loss_cls = epoch_loss_cls / num_batches
        mean_loss_angle = epoch_loss_angle / num_batches
        mean_loss_iou = epoch_loss_iou / num_batches

        log_dict = {
            "mean_val_loss": mean_val_loss,
            "mean_loss": mean_loss,
            "mean_loss_cls": mean_loss_cls,
            "mean_loss_angle": mean_loss_angle,
            "mean_loss_iou": mean_loss_iou
        }
        wandb.log(log_dict)

        if (epoch + 1) % save_interval == 0:
            if not osp.exists(model_dir):
                os.makedirs(model_dir)

            ckpt_fpath = osp.join(model_dir, 'latest.pth')
            torch.save({ 
                        'epoch': epoch+1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_validation_loss': best_val_loss,
                        }, ckpt_fpath)

    print("***Training Finished!***")

def main(args):
    do_training(**args.__dict__)

if __name__ == '__main__':
    args = parse_args()
    main(args)