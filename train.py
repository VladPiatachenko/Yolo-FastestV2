import os
import math
import time
import argparse
import numpy as np
from tqdm import tqdm
from numpy.testing._private.utils import print_assert_equal

import torch
from torch import optim
from torch.utils.data import dataset
from numpy.core.fromnumeric import shape

from torchsummary import summary

import utils.loss
import utils.utils
import utils.datasets
from model.detector import Detector
import re

# Define a function to extract the starting epoch from the model file name
def extract_start_epoch(model_path):
    filename = os.path.basename(model_path)
    match = re.search(r'-(\d+)-epoch-', filename)
    if match:
        return int(match.group(1))
    else:
        return 0

if __name__ == '__main__':
    # Specify the training configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='',
                        help='Specify the training profile *.data')
    parser.add_argument('--resume', action='store_true',
                        help='Whether to resume training from a saved model')
    parser.add_argument('--model_path', type=str, default='',
                        help='Path to the saved model file (.pth)')
    opt = parser.parse_args()
    cfg = utils.utils.load_datafile(opt.data)

    print("Training configuration:")
    print(cfg)
    
    # Data loading
    train_dataset = utils.datasets.TensorDataset(cfg["train"], cfg["width"], cfg["height"], imgaug=True)
    val_dataset = utils.datasets.TensorDataset(cfg["val"], cfg["width"], cfg["height"], imgaug=False)

    batch_size = int(cfg["batch_size"] / cfg["subdivisions"])
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    # Training dataset
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   collate_fn=utils.datasets.collate_fn,
                                                   num_workers=nw,
                                                   pin_memory=True,
                                                   drop_last=True,
                                                   persistent_workers=True
                                                   )
    # Validation dataset
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 collate_fn=utils.datasets.collate_fn,
                                                 num_workers=nw,
                                                 pin_memory=True,
                                                 drop_last=False,
                                                 persistent_workers=True
                                                 )

    # Specify the backend device (CUDA or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load saved model if resuming training
    if opt.resume and opt.model_path:
        start_epoch = extract_start_epoch(opt.model_path)
        model = Detector(cfg["classes"], cfg["anchor_num"], load_param=True)
        model = model.to(device)
        model.load_state_dict(torch.load(opt.model_path))
        print("Resuming training from epoch %d, model: %s" % (start_epoch, opt.model_path))
    else:
        start_epoch = 0
        print("Starting training from scratch")
        model = Detector(cfg["classes"], cfg["anchor_num"], load_param=False)
        model = model.to(device)
        load_param = False  # Set load_param to False when starting from scratch

    # Initialize the model structure
    summary(model, input_size=(3, cfg["height"], cfg["width"]))

    # Build the SGD optimizer
    optimizer = optim.SGD(params=model.parameters(),
                          lr=cfg["learning_rate"],
                          momentum=0.949,
                          weight_decay=0.0005,
                          )

    # Learning rate decay strategy
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=cfg["steps"],
                                               gamma=0.1)

    print('Starting training for %g epochs...' % (cfg["epochs"] - start_epoch))

    batch_num = 0
    best_ap = 0.0  # Track the best average precision
    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        pbar = tqdm(train_dataloader)

        for imgs, targets in pbar:
            # Data preprocessing
            imgs = imgs.to(device).float() / 255.0
            targets = targets.to(device)

            # Model inference
            preds = model(imgs)
            # Loss calculation
            iou_loss, obj_loss, cls_loss, total_loss = utils.loss.compute_loss(preds, targets, cfg, device)

            # Backpropagation to compute gradients
            total_loss.backward()

            # Learning rate warm-up
            for g in optimizer.param_groups:
                warmup_num =  5 * len(train_dataloader)
                if batch_num <= warmup_num:
                    scale = math.pow(batch_num/warmup_num, 4)
                    g['lr'] = cfg["learning_rate"] * scale

                lr = g["lr"]

            # Update model parameters
            if batch_num % cfg["subdivisions"] == 0:
                optimizer.step()
                optimizer.zero_grad()

            # Print relevant information
            info = "Epoch:%d LR:%f CIou:%f Obj:%f Cls:%f Total:%f" % (
                    epoch, lr, iou_loss, obj_loss, cls_loss, total_loss)
            pbar.set_description(info)

            batch_num += 1

        # Save the model
        if epoch % 3 == 0 and epoch > 0:
            model.eval()
            # Model evaluation
            print("Compute mAP...")
            _, _, AP, _ = utils.utils.evaluation(val_dataloader, cfg, model, device)
            print("Compute PR...")
            precision, recall, _, f1 = utils.utils.evaluation(val_dataloader, cfg, model, device, 0.3)
            print("Precision:%f Recall:%f AP:%f F1:%f"%(precision, recall, AP, f1))

            # Save model to Google Drive directory
            model_save_path = f"/content/drive/MyDrive/checkpoints/{cfg['model_name']}-{epoch}-epoch-{AP:.6f}ap-model.pth"
            torch.save(model.state_dict(), model_save_path)

            # Update best model if current model has better AP
            if AP > best_ap:
                best_ap = AP
                best_model_path = model_save_path

        # Adjust learning rate
        scheduler.step()
