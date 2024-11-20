from __future__ import print_function
import os
import torch
from torch import nn
from torch.utils import data
from torch.nn import DataParallel
from torch.optim.lr_scheduler import StepLR

import numpy as np
from tqdm import tqdm

from models import resnet_face18
from models import (
    AddMarginProduct,
    ArcMarginProduct, 
    SphereProduct,
    FocalLoss,
    HingleLoss
)
from utils import Visualizer, view_model
from config import Config
from datasets import Dataset, custom_dataset
from test import get_lfw_list, lfw_test



def save_model(model, optimizer, metric_fc, save_path, name, iter_cnt):
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    save_name = os.path.join(save_path, name + '_last.pth')
    torch.save({
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "epoch": iter_cnt,
        "metric_fc": metric_fc.state_dict()
    }, save_name)
    return save_name


if __name__ == '__main__':
    opt = Config()
    if opt.display:
        visualizer = Visualizer()
    
    device = torch.device("cuda")
    train_dataset = custom_dataset(opt.train_root, input_shape=opt.input_shape)
    trainloader = data.DataLoader(train_dataset,
                                  batch_size=opt.train_batch_size,
                                  shuffle=True,
                                  num_workers=opt.num_workers)

    # identity_list = get_lfw_list(opt.lfw_test_list)
    # img_paths = [os.path.join(opt.lfw_root, each) for each in identity_list]

    print('{} train iters per epoch:'.format(len(trainloader)))

    if opt.loss == 'focal_loss':
        criterion = FocalLoss(gamma=2)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if opt.use_hingle_loss:
      svm_hinge_loss = HingleLoss(delta=opt.delta)


    if opt.backbone == 'resnet18':
        model = resnet_face18(use_se=opt.use_se)
    else:
        raise "Not Implemented"

    if opt.metric == 'add_margin':
        metric_fc = AddMarginProduct(512, opt.num_classes, s=30, m=0.35)
    elif opt.metric == 'arc_margin':
        metric_fc = ArcMarginProduct(512, opt.num_classes, s=30, m=0.5, easy_margin=opt.easy_margin)
    elif opt.metric == 'sphere':
        metric_fc = SphereProduct(512, opt.num_classes, m=4)
    else:
        metric_fc = nn.Linear(512, opt.num_classes)

    
    if opt.optimizer == 'sgd':
        optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                    lr=opt.lr, weight_decay=opt.weight_decay)
    else:
        optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                     
                                     lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = StepLR(optimizer, step_size=opt.lr_step, gamma=0.1)

    start_epoch = 0
    if opt.resume is not None:
        checkpoint = torch.load(opt.resume, map_location ="cpu",
                                weights_only=True)

        state_dict = checkpoint["model"]
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k,
                          v in checkpoint['model'].items()}
            
        model.load_state_dict(state_dict, strict=True)
        optimizer.load_state_dict(checkpoint["optim"])
        start_epoch = checkpoint["epoch"]

        scheduler.last_epoch = start_epoch
        start_epoch += 1

    model.to(device)
    model = DataParallel(model)
    metric_fc.to(device)
    metric_fc = DataParallel(metric_fc)


    for i in range(start_epoch, opt.max_epoch):
        model.train()
        scheduler.step()

        max_iter = len(trainloader)
        pbar = tqdm(enumerate(trainloader), total=max_iter)
        for ii, pair in pbar:
            data_input, label = pair
            data_input = data_input.to(device)
            label = torch.Tensor(label).to(device).long()
            feature = model(data_input)
            output = metric_fc(feature, label)
            loss = criterion(output, label) 
            if opt.use_hingle_loss:
              loss += opt.hingle_weight * svm_hinge_loss(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            output = output.data.cpu().numpy()
            output = np.argmax(output, axis=1)
            label = label.data.cpu().numpy()
            acc = np.mean((output == label).astype(int))
            pbar.set_description(
              f"epoch {i} loss {loss.item():.4f} acc {acc:.6f}"
            )

        save_model(model, optimizer, metric_fc, opt.checkpoints_path, opt.backbone, i)

        # model.eval()
        # acc = lfw_test(model, img_paths, identity_list, opt.lfw_test_list, opt.test_batch_size)
        # if opt.display:
        #     visualizer.display_current_results(iters, acc, name='test_acc')