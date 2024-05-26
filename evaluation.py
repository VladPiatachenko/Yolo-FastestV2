import os
import torch
import argparse
from tqdm import tqdm
from torchsummary import summary

import utils.utils
import utils.datasets
import model.detector

if __name__ == '__main__':
    # 指定训练配置文件
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='',
                        help='Specify training profile *.data')
    parser.add_argument('--weights', type=str, default='',
                        help='The path of the model')
    opt = parser.parse_args()
    cfg = utils.utils.load_datafile(opt.data)

    assert os.path.exists(opt.weights), "请指定正确的模型路径"

    # 打印消息
    print("评估配置:")
    print("model_name:%s" % cfg["model_name"])
    print("width:%d height:%d" % (cfg["width"], cfg["height"]))
    print("val:%s" % (cfg["val"]))
    print("model_path:%s" % (opt.weights))

    # 加载数据
    val_dataset = utils.datasets.TensorDataset(cfg["val"], cfg["width"], cfg["height"], imgaug=False)

    batch_size = int(cfg["batch_size"] / cfg["subdivisions"])
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])

    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 collate_fn=utils.datasets.collate_fn,
                                                 num_workers=nw,
                                                 pin_memory=True,
                                                 drop_last=False,
                                                 persistent_workers=True
                                                 )

    # 指定后端设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型
    model = model.detector.Detector(cfg["classes"], cfg["anchor_num"], True).to(device)

    def adjust_state_dict_keys(state_dict, prefix_to_remove='module.'):
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith(prefix_to_remove):
                new_key = k[len(prefix_to_remove):]
            else:
                new_key = k
            new_state_dict[new_key] = v
        return new_state_dict

    # Load the state dictionary
    checkpoint = torch.load(opt.weights, map_location=device)

    # Adjust the keys if necessary
    if 'model_state_dict' in checkpoint:
        state_dict = adjust_state_dict_keys(checkpoint['model_state_dict'])
    else:
        state_dict = adjust_state_dict_keys(checkpoint)

    # Load the model state dictionary
    model.load_state_dict(state_dict, strict=False)
    
    # 设置模型为评估模式
    model.eval()

    # 打印模型结构
    summary(model, input_size=(3, cfg["height"], cfg["width"]))

    # 模型评估
    print("computing mAP...")
    _, _, AP, _ = utils.utils.evaluation(val_dataloader, cfg, model, device)
    print("computing PR...")
    precision, recall, _, f1 = utils.utils.evaluation(val_dataloader, cfg, model, device, 0.3)
    print("Precision:%f Recall:%f AP:%f F1:%f" % (precision, recall, AP, f1))
