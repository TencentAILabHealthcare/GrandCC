import numpy as np
import torch
import random
import torch.nn as nn
import torch.backends.cudnn as cudnn
import argparse
import logging
from torch_geometric.utils import to_dense_batch
from torch_geometric.data import DataLoader
from sklearn.model_selection import StratifiedKFold

import torch.nn.functional as F

from utils import *
from model_GAT_v4 import *


def test(opt, model, test_dataset):

    model.eval()
    test_loader = DataLoader(dataset=test_dataset, batch_size=opt.batch_size, shuffle=False)

    probs_all, gt_all = None, np.array([])
    loss_test, grad_acc_test = 0, 0

    for batch_idx, (data) in enumerate(test_loader):

        te_features, te_fc_features, te_preds, _, _ = model(
            data.x.cuda(), data.edge_index.cuda(), data.y.cuda(), data.batch.cuda(), opt)

        if batch_idx == 0:
            features_all = te_features.detach().cpu().numpy()
            fc_features_all = te_fc_features.detach().cpu().numpy()
        else:
            features_all = np.concatenate((features_all, te_features.detach().cpu().numpy()), axis=0)
            fc_features_all = np.concatenate((fc_features_all, te_fc_features.detach().cpu().numpy()), axis=0)

        loss_reg = define_reg(model)
        loss_func = nn.CrossEntropyLoss()
        grad_loss = loss_func(te_preds, data.y.cuda())

        loss = opt.lambda_nll * grad_loss + opt.lambda_reg * loss_reg
        loss_test += loss.data.item()

        gt_all = np.concatenate((gt_all, data.y.reshape(-1)))  # Logging Information

        pred = te_preds.argmax(dim=1, keepdim=True)
        grad_acc_test += pred.eq(data.y.cuda().view_as(pred)).sum().item()
        probs_np = te_preds.detach().cpu().numpy()
        probs_all = probs_np if probs_all is None else np.concatenate((probs_all, probs_np), axis=0)  # Logging Information

    loss_test /= len(test_loader.dataset)
    grad_acc_test = grad_acc_test / len(test_loader.dataset)
    pred_test = [probs_all, gt_all]

    return loss_test, grad_acc_test, pred_test, features_all, fc_features_all

def external_test(opt, model, test_dataset):

    model.eval()
    test_loader = DataLoader(dataset=test_dataset, batch_size=opt.batch_size, shuffle=False)
    probs_all = None

    for batch_idx, (data) in enumerate(test_loader):

        te_features, te_fc_features, te_preds, _, _ = model(
            data.x.cuda(), data.edge_index.cuda(), data.y.cuda(), data.batch.cuda(), opt)
        # print("features:", te_features)
        probs_np = F.softmax(te_preds, 1).detach().cpu().numpy()
        probs_all = probs_np if probs_all is None else np.concatenate((probs_all, probs_np), axis=0)

    return probs_all

