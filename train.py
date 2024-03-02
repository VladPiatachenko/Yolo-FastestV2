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

if __name__ == '__main__':
    # Specify the training configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='', help='Specify training profile *.data')
    parser.add_argument('--checkpoint', type=str, default='',
                        help='Path to the checkpoint file to continue training from')
    opt = parser.parse_args()
    cfg = utils.utils.load_datafile(opt.data)

    # Define checkpoint interval (e.g., save checkpoint every 2 epochs)
    checkpoint_interval = 2
    checkpoint_dir = '/content/drive/My Drive/checkpoints'

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
                                  persistent_workers=True
                                  )
    # Validation set
    val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                collate_fn=utils.datasets.collate_fn,
                                num_workers=nw,
                                pin_memory=True,
                                drop_last=False,
                                persistent_workers=True
                                )

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

    # Load the parameters of the pre-trained model
    if load_param:
        model.load_state_dict(torch.load(premodel_path, map_location=device), strict=False)
        print("Loaded fine-tuned model parameters from: %s" % premodel_path)
    else:
        print("Initialize weights from: model/backbone/backbone.pth")

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

    print('Starting training for %g epochs...' % cfg["epochs"])

    # Initialize epoch to 0 if it's not found in local variables
    epoch = 0

    # Initialize batch_num
    batch_num = 0

    # Start from the specified epoch or the next epoch after the loaded checkpoint
    start_epoch = epoch + 1 if 'epoch' in locals() else 0

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

                # Backpropagation to compute gradients
                total_loss.backward()

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

                # Save checkpoints
                if epoch % checkpoint_interval == 0:
                    checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pth")
                    torch.save({
                        'epoch': epoch,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }, checkpoint_path)
                    print(f"Checkpoint saved at {checkpoint_path}")

            # Save the model
            if epoch % 10 == 0 and epoch > 0:
                model.eval()
                # Model evaluation
                print("Compute mAP...")
                _, _, AP, _ = utils.utils.evaluation(val_dataloader, cfg, model, device)
                print("Compute PR...")
                precision, recall, _, f1 = utils.utils.evaluation(val_dataloader, cfg, model, device, 0.3)
                print("Precision:%f Recall:%f AP:%f F1:%f" % (precision, recall, AP, f1))

                torch.save(model.state_dict(), "weights/%s-%d-epoch-%fap-model.pth" %
                           (cfg["model_name"], epoch, AP))

            # Learning rate adjustment
            scheduler.step()

    except KeyboardInterrupt:
        print("Training interrupted. Saving checkpoint...")
        # Save checkpoint or any other necessary clean-up
