import os
import os.path as osp
import time
import math
from datetime import timedelta
from argparse import ArgumentParser

import torch
import numpy as np
import random
from torch import cuda
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torch.optim import lr_scheduler
from tqdm import tqdm

from east_dataset import EASTDataset
from dataset import SceneTextDataset
from model import EAST

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', '../data/medical'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR',
                                                                        'trained_models'))

    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--image_size', type=int, default=2048)
    parser.add_argument('--input_size', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=2)
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--ignore_tags', type=list, default=['masked', 'excluded-region', 'maintable', 'stamp'])
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')
    return args


def do_training(data_dir, model_dir, device, image_size, input_size, num_workers, batch_size,
                learning_rate, max_epoch, save_interval, ignore_tags, seed):
    seed_everything(seed)
    
    dataset = SceneTextDataset(
        data_dir,
        split='train',
        image_size=image_size,
        crop_size=input_size,
        ignore_tags=ignore_tags
    )
    dataset = EASTDataset(dataset)
    
    train_dataset, val_dataset = random_split(dataset, [80,20])
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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch // 2], gamma=0.1)

    best_val_loss = np.inf
    for epoch in range(max_epoch):
        train_start = time.time()
        model.train()
        train_loss = 0
        val_loss = 0
        print()
        
        with tqdm(total=num_batches) as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
                pbar.set_description('[Epoch {}]'.format(epoch + 1)) # 진행 바 왼쪽에 Epoch 진행 상황 추가

                loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                pbar.update(1) # 수동으로 진행률을 1씩 증가 시킨다
                train_dict = {
                    '(Train)Class loss': extra_info['cls_loss'], '(Train)Angle loss': extra_info['angle_loss'],
                    '(Train)IoU loss': extra_info['iou_loss']
                }
                pbar.set_postfix(train_dict) # 진행 바 오른쪽에 Train Loss 설명 추가

                mean_train_loss = train_loss / num_batches

            with torch.no_grad():
                val_start = time.time()
                model.eval()      
                print("\nEvaluating validation results...")
                for img, gt_score_map, gt_geo_map, roi_mask in val_loader:
                    loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                    val_loss += loss.item()

                mean_val_loss = val_loss / num_batches
                if best_val_loss > mean_val_loss:
                    best_val_loss = mean_val_loss
                    best_val_loss_epoch = epoch+1

                print(f'(Val) Class loss={extra_info["cls_loss"]}, (Val) Angle loss={extra_info["angle_loss"]}, (Val) IoU loss={extra_info["iou_loss"]}')

            print('(Train) Mean loss: {:.4f} | Elapsed time: {}'.format(mean_train_loss, timedelta(seconds=time.time() - train_start)))
            print('(Val)   Mean loss: {:.4f} | Elapsed time: {}'.format(mean_val_loss, timedelta(seconds=time.time() - val_start)))
            print('Best Validation Loss: {:.4f} at Epoch {}'.format(best_val_loss, best_val_loss_epoch))
            

        scheduler.step()

        if (epoch + 1) % save_interval == 0:
            if not osp.exists(model_dir):
                os.makedirs(model_dir)

            ckpt_fpath = osp.join(model_dir, 'latest.pth')
            torch.save(model.state_dict(), ckpt_fpath)

    print("***Training Finished!***")

def main(args):
    do_training(**args.__dict__)

if __name__ == '__main__':
    args = parse_args()
    main(args)