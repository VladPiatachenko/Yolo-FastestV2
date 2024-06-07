import os
import math
import time
import argparse
import numpy as np
from tqdm import tqdm
import torch
from torch import optim
from torch.utils.data import dataset
from torchsummary import summary
import utils.loss
import utils.utils
import utils.datasets
from model.detector import Detector
import re
from datetime import datetime  # Import datetime module for timestamp

# Define a function to extract the starting epoch from the model file name
def extract_start_epoch(model_path):
    filename = os.path.basename(model_path)
    match = re.search(r'-(\d+)-epoch-', filename)
    if match:
        return int(match.group(1))
    else:
        return 0
        
# Define a function for label smoothing
def smooth_labels(labels, epsilon, num_classes):
    smoothed_labels = labels * (1 - epsilon) + epsilon / num_classes
    return smoothed_labels

if __name__ == '__main__':
    # Specify the training configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='',
                        help='Specify the training profile *.data')
    parser.add_argument('--resume', action='store_true',
                        help='Whether to resume training from a saved model')
    parser.add_argument('--model_path', type=str, default='',
                        help='Path to the saved model file (.pth)')
    parser.add_argument('--save_location', type=str, default='',
                        help='Path to save the best model checkpoint')
    opt = parser.parse_args()
    cfg = utils.utils.load_datafile(opt.data)
    
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
    if (opt.resume and opt.model_path):
        start_epoch = extract_start_epoch(opt.model_path)
        model = Detector(cfg["classes"], cfg["anchor_num"], load_param=True)
        model = model.to(device)
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(opt.model_path))
        else:
            model.load_state_dict(torch.load(opt.model_path, map_location=torch.device('cpu')))
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
    best_model_path = ""  # Path to save the best model
    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        pbar = tqdm(train_dataloader)

        for imgs, targets in pbar:
            # Data preprocessing
            imgs = imgs.to(device).float() / 255.0
            targets = targets.to(device)
            
            # Print original targets before smoothing
            print(f"Original targets: {targets[:2]}")  # Print first two target sets for brevity

            # Apply label smoothing
            targets[..., 5:] = smooth_labels(targets[..., 5:], epsilon=0.1, num_classes=cfg["classes"])
            
            # Print smoothed targets
            print(f"Smoothed targets: {targets[:2]}")  # Print first two smoothed target sets for brevity

            # Model inference
            preds = model(imgs)
            # Loss calculation
            iou_loss, obj_loss, cls_loss, total_loss = utils.loss.compute_loss(preds, targets, cfg, device)

            # Print loss values
            print(f"Loss values - IoU Loss: {iou_loss}, Obj Loss: {obj_loss}, Cls Loss: {cls_loss}, Total Loss: {total_loss}")

            # Backpropagation to compute gradients
            total_loss.backward()

            # Learning rate warm-up
            for g in optimizer.param_groups:
                warmup_num =  5 * len(train_dataloader)
                if batch_num <= warmup_num:
                    scale = math.pow(batch_num / warmup_num, 4)
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

        # Save the model if it has the best AP
        model.eval()
        # Model evaluation
        print("Compute mAP...")
        _, _, AP, _ = utils.utils.evaluation(val_dataloader, cfg, model, device)
        print("Compute PR...")
        precision, recall, _, f1 = utils.utils.evaluation(val_dataloader, cfg, model, device, 0.3)
        print("Precision:%f Recall:%f AP:%f F1:%f"%(precision, recall, AP, f1))

        # Update best model if current model has better AP
        if AP > best_ap:
            best_ap = AP
            current_date = datetime.now().strftime('%Y-%m-%d')
            best_model_path = f"{opt.save_location}/{cfg['model_name']}-best-model-{epoch}-epoch-{current_date}.pth"
            torch.save(model.state_dict(), best_model_path)
            print(f"Checkpoint saved at: {best_model_path}")
        
        # Adjust learning rate
        scheduler.step()

    print(f"Best model saved at: {best_model_path}")
