import torch
import torch.nn as nn
import numpy as np

layer_index = [0, 0, 0, 1, 1, 1]

def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False):
    box2 = box2.t()

    if x1y1x2y2:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = (w1 * h1 + 1e-16) + w2 * h2 - inter

    iou = inter / union

    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)

        if GIoU:
            c_area = cw * ch + 1e-16
            return iou - (c_area - union) / c_area
        if DIoU or CIoU:
            c2 = cw ** 2 + ch ** 2 + 1e-16
            rho2 = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + \
                   ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4

            if DIoU:
                return iou - rho2 / c2
            elif CIoU:
                v = (4 / np.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                alpha = v / (1 - iou + v)
                return iou - (rho2 / c2 + v * alpha)

    return iou

def build_target(preds, targets, cfg, device):
    # Implementation of build_target function
    pass

def smooth_BCE(eps=0.1):
    return 1.0 - 0.5 * eps, 0.5 * eps

def compute_loss(preds, targets, cfg, device):
    lbox, lobj, lcls = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

    for i, pred in enumerate(preds):
        if i % 3 == 0:
            lbox += compute_localization_loss(pred, targets, cfg, device)
        elif i % 3 == 1:
            lobj += compute_objectness_loss(pred, targets, cfg, device)
        elif i % 3 == 2:
            lcls += compute_classification_loss(pred, targets, cfg, device)
        else:
            raise ValueError("Invalid prediction index.")

    lbox *= cfg["box_loss_weight"]
    lobj *= cfg["obj_loss_weight"]
    lcls *= cfg["cls_loss_weight"]

    loss = lbox + lobj + lcls
    return lbox, lobj, lcls, loss

def smooth_labels(true_labels, smooth_factor=0.05):
    num_classes = true_labels.size(-1)
    smooth_labels = true_labels * (1.0 - smooth_factor)
    smooth_labels += smooth_factor / num_classes
    return smooth_labels

def compute_localization_loss(pred, targets, cfg, device):
    # Placeholder for localization loss computation
    return torch.tensor(0.0, device=device)

def compute_objectness_loss(pred, targets, cfg, device):
    # Placeholder for objectness loss computation
    return torch.tensor(0.0, device=device)

def compute_classification_loss(pred, targets, cfg, device):
    # Placeholder for classification loss computation
    return torch.tensor(0.0, device=device)
