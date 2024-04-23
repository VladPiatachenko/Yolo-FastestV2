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
    tcls, tbox, indices, anch = [], [], [], []
    #anchor box数量, 当前batch的标签数量
    anchor_num, label_num = cfg["anchor_num"], targets.shape[0]

    #加载anchor配置
    anchors = np.array(cfg["anchors"])
    anchors = torch.from_numpy(anchors.reshape(len(preds) // 3, anchor_num, 2)).to(device)
    
    gain = torch.ones(7, device = device)

    at = torch.arange(anchor_num, device = device).float().view(anchor_num, 1).repeat(1, label_num)
    targets = torch.cat((targets.repeat(anchor_num, 1, 1), at[:, :, None]), 2)

    g = 0.5  # bias
    off = torch.tensor([[0, 0],
                        [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                        # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                        ], device = device).float() * g  # offsets

    for i, pred in enumerate(preds):
        if i % 3 == 0:
            #输出特征图的维度
            _, _, h, w = pred.shape

            assert cfg["width"]/w == cfg["height"]/h, "特征图宽高下采样不一致"
            
            #计算下采样倍数
            stride = cfg["width"]/w

            #该尺度特征图对应的anchor配置
            anchors_cfg = anchors[layer_index[i]]/stride

            #将label坐标映射到特征图上
            gain[2:6] = torch.tensor(pred.shape)[[3, 2, 3, 2]]

            gt = targets * gain 

            if label_num:
                #anchor iou匹配
                r = gt[:, :, 4:6] / anchors_cfg[:, None]
                j = torch.max(r, 1. / r).max(2)[0] < 2

                t = gt[j]
                #扩充维度并复制数据
                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors_cfg[a])  # anchors
            tcls.append(c)  # class

    return tcls, tbox, indices, anch

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

    lbox *= 1.0#cfg["box_loss_weight"]
    lobj *= 1.0#cfg["obj_loss_weight"]
    lcls *= 1.0#cfg["cls_loss_weight"]

    loss = lbox + lobj + lcls
    return lbox, lobj, lcls, loss

def smooth_labels(true_labels, smooth_factor=0.05):
    num_classes = true_labels.size(-1)
    smooth_labels = true_labels * (1.0 - smooth_factor)
    smooth_labels += smooth_factor / num_classes
    return smooth_labels

def compute_localization_loss(pred, targets, cfg, device):
    lbox = torch.tensor(0.0, device=device)  # Initialize localization loss

    for i, pred_batch in enumerate(pred):
        for j, pred_anchor in enumerate(pred_batch):
            target_mask = targets[:, :, 0] == i  # Filter targets for the current batch index
            if target_mask.any():
                pred_box = pred_anchor[target_mask[:, j]]  # Predicted bounding boxes for targets in this batch
                target_box = targets[target_mask][:, j, 2:]  # Target bounding boxes
                lbox += F.mse_loss(pred_box, target_box, reduction='sum')  # Compute MSE loss

    return lbox


def compute_objectness_loss(pred, targets, cfg, device):
    lobj = torch.tensor(0.0, device=device)  # Initialize objectness loss

    for i, pred_batch in enumerate(pred):
        for j, pred_anchor in enumerate(pred_batch):
            target_mask = targets[:, :, 0] == i  # Filter targets for the current batch index
            if target_mask.any():
                obj_pred = pred_anchor[target_mask]  # Predicted objectness scores
                obj_target = targets[target_mask, 1]  # Target objectness scores
                lobj += F.binary_cross_entropy_with_logits(obj_pred, obj_target, reduction='sum')  # Binary cross-entropy loss

    return lobj

def compute_classification_loss(pred, targets, cfg, device):
    lcls = torch.tensor(0.0, device=device)  # Initialize classification loss

    for i, pred_batch in enumerate(pred):
        for j, pred_anchor in enumerate(pred_batch):
            target_mask = targets[:, :, 0] == i  # Filter targets for the current batch index
            if target_mask.any():
                class_pred = pred_anchor[target_mask]  # Predicted class scores
                class_target = targets[target_mask, 1].long()  # Target class indices
                lcls += F.cross_entropy(class_pred, class_target, reduction='sum')  # Cross-entropy loss

    return lcls
