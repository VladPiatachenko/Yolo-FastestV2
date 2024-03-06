import os
import math
import time
import argparse
import numpy as np
from tqdm import tqdm
import torch
from torch import optim
from torch.utils.data import DataLoader
from torchsummary import summary
import utils.loss
import utils.utils
import utils.datasets
import model.detector
import re

if __name__ == '__main__':
    # Specify the training configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='', help='Specify training profile *.data')
    parser.add_argument('--checkpoint', type=str, default='', help='Path to the checkpoint file to continue training from')
    opt = parser.parse_args()
    cfg = utils.utils.load_datafile(opt.data)

    # Define checkpoint interval (e.g., save checkpoint every 2 epochs)
    checkpoint_interval = 2
    checkpoint_dir = '/content/drive/My Drive/checkpoints'
    checkpoint_model_dir = '/content/drive/My Drive/checkpoints/weights'

    print("Training configuration:")
    print(cfg)

    # Create the checkpoint directory if it does not exist
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Load the dataset
    train_dataset = utils.datasets.TensorDataset(cfg["train"], cfg["width"], cfg["height"], imgaug=True)
    val_dataset = utils.datasets.TensorDataset(cfg["val"], cfg["width"], cfg["height"], imgaug=False)

    batch_size = int(cfg["batch_size"] / cfg["subdivisions"])
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])

    # Training set
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  collate_fn=utils.datasets.collate_fn,
                                  num_workers=nw,
                                  pin_memory=True,
                                  drop_last=True,
                                  persistent_workers=True)
    # Validation set
    val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                collate_fn=utils.datasets.collate_fn,
                                num_workers=nw,
                                pin_memory=True,
                                drop_last=False,
                                persistent_workers=True)

    # Specify the device (CUDA or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Check if a pre-trained model should be loaded
    load_param = False
    premodel_path = cfg["pre_weights"]
    if premodel_path is not None and os.path.exists(premodel_path):
        load_param = True

    # Initialize the model structure
    model = model.detector.Detector(cfg["classes"], cfg["anchor_num"], load_param).to(device)
    summary(model, input_size=(3, cfg["height"], cfg["width"]))

    # Build the SGD optimizer
    optimizer = optim.SGD(params=model.parameters(),
                          lr=cfg["learning_rate"],
                          momentum=0.949,
                          weight_decay=0.0005)

    # Load the parameters of the pre-trained model
    if load_param:
        print("Loading pre-trained model parameters from:", premodel_path)
        model_state_dict = torch.load(premodel_path, map_location=device)
        model.load_state_dict(model_state_dict, strict=False)
        print("Model state loaded successfully.")
        print("Model parameters:")
        for name, param in model.named_parameters():
            print(name, param.size())

        # Print optimizer state before loading
        print("Optimizer state before loading checkpoint:")
        for param_group in optimizer.param_groups:
            print("Learning rate:", param_group['lr'])

        optimizer.load_state_dict(model_state_dict['optimizer'])

        # Print optimizer state after loading
        print("Optimizer state after loading checkpoint:")
        for param_group in optimizer.param_groups:
            print("Learning rate:", param_group['lr'])
    else:
        print("Initialize weights from: model/backbone/backbone.pth")

    # Learning rate decay strategy
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=cfg["steps"],
                                               gamma=0.1)

    print('Starting training for %g epochs...' % cfg["epochs"])

    # Initialize epoch to 0 if it's not found in local variables
    epoch = 0

    # Initialize batch_num
    batch_num = 0

    # Start from the specified epoch or the next epoch after the loaded checkpoint
    start_epoch = epoch + 1 if 'epoch' in locals() else 0

    # Extract the starting epoch number from the checkpoint filename
    if opt.checkpoint:
        start_epoch_match = re.search(r'model_epoch_(\d+)\.pth', opt.checkpoint)
        if start_epoch_match:
            start_epoch = int(start_epoch_match.group(1))
            print("Starting epoch extracted from checkpoint filename:", start_epoch)

    try:
        for epoch in range(start_epoch, cfg["epochs"]):
            model.train()
            pbar = tqdm(train_dataloader)

            for imgs, targets in pbar:
                # Data preprocessing
                imgs = imgs.to(device).float() / 255.0
                targets = targets.to(device)

                # Model inference
                preds = model(imgs)
                # Compute loss
                iou_loss, obj_loss, cls_loss, total_loss = utils.loss.compute_loss(preds, targets, cfg, device)

                print("Loss values before backpropagation - Epoch:%d Batch:%d - CIou:%f Obj:%f Cls:%f Total:%f" % (
                    epoch, batch_num, iou_loss, obj_loss, cls_loss, total_loss))

                # Backpropagation to compute gradients
                total_loss.backward()

                print("Loss values after backpropagation - Epoch:%d Batch:%d - CIou:%f Obj:%f Cls:%f Total:%f" % (
                    epoch, batch_num, iou_loss, obj_loss, cls_loss, total_loss))

                # Warmup for learning rate
                for g in optimizer.param_groups:
                    warmup_num = 5 * len(train_dataloader)
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

            # Save checkpoints at the end of the epoch
            if epoch % checkpoint_interval == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pth")
                checkpoint = {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'pre_weights': cfg['pre_weights']
                }
                torch.save(checkpoint, checkpoint_path)
                print(f"Checkpoint saved at {checkpoint_path}")

                # Save the model
                if epoch % 10 == 0 and epoch > 0:
                    print("Saveing the model...")
                    model.eval()
                    # Model evaluation
                    print("Compute mAP...")
                    _, _, AP, _ = utils.utils.evaluation(val_dataloader, cfg, model, device)
                    print("Compute PR...")
                    precision, recall, _, f1 = utils.utils.evaluation(val_dataloader, cfg, model, device, 0.3)
                    print("Precision:%f Recall:%f AP:%f F1:%f" % (precision, recall, AP, f1))
                  # Create the directory if it doesn't exist
                    os.makedirs(checkpoint_dir, exist_ok=True)
                
                    # Save the model checkpoint to the specified directory
                    checkpoint_path = os.path.join(checkpoint_model_dir, "%s-%d-epoch-%fap-model.pth" % (cfg["model_name"], epoch, AP))
                    torch.save(model.state_dict(), checkpoint_path)


                # Learning rate adjustment
                scheduler.step()

    except KeyboardInterrupt:
        print("Training interrupted. Saving checkpoint...")
        # Save checkpoint or any other necessary clean-up
