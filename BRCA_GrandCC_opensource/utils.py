import numpy as np
import math
import torch
import torch.nn as nn
import pandas as pd
import torch.utils.data as Data
from torch.utils.data.dataset import Dataset

## for semi-supervised sampler
from torch.utils.data.sampler import Sampler
import itertools

import lifelines
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test

from sklearn.metrics import accuracy_score, balanced_accuracy_score, \
    precision_score, recall_score


################
# Data Utils
################

def cal_sample_weight(labels, num_class, use_sample_weight=True):
    """ calculate sample weights based on training label distribution
    """
    if not use_sample_weight:
        return np.ones(len(labels)) / len(labels)
    class_samples = [100, 100, 100, 100, 100]

    sample_weight = np.zeros(labels.shape)
    for i in range(num_class):
        sample_weight[np.where(labels == i)[0]] = 517/class_samples[i]

    return sample_weight


def one_hot_tensor(y, num_dim):
    """convet y to one hot coding"""
    y_onehot = torch.zeros(y.shape[0], num_dim)
    y_onehot.scatter_(1, y.view(-1, 1), 1)

    return y_onehot


################
# Grading Utils
################
def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def print_model(model, optimizer):
    print(model)
    print("Model's state_dict:")
    # Print model's state_dict
    for param_tensor in model.state_dict():
        print(param_tensor,"\t", model.state_dict()[param_tensor].size())
    print("optimizer's state_dict:")
    # Print optimizer's state_dict
    for var_name in optimizer.state_dict():
        print(var_name,"\t", optimizer.state_dict()[var_name])


def init_max_weights(module):
    for m in module.modules():
        if type(m) == nn.Linear:
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            m.bias.data.zero_()


def compute_ROC_AUC(test_pred, gt_labels):
    labels_oh = np.zeros(shape=(gt_labels.shape[0], 2))
    labels_oh[np.arange(0, gt_labels.shape[0]), gt_labels] = 1
    fpr, tpr, thresh = roc_curve(labels_oh.ravel(), test_pred.ravel())
    aucroc = auc(fpr, tpr)

    return aucroc


def compute_metrics(test_pred, gt_labels):

    labels_oh = np.zeros(shape=(gt_labels.shape[0], test_pred.shape[1]))
    labels_oh[np.arange(0, gt_labels.shape[0]), gt_labels] = 1

    idx = np.argmax(test_pred, axis=1)
    pred = np.zeros(shape=(idx.shape[0], test_pred.shape[1]))
    pred[np.arange(0, idx.shape[0]), idx] = 1

    print(labels_oh.shape, pred.shape)

    macro_f1_score = f1_score(labels_oh, pred, average='macro')
    precision = precision_score(labels_oh, pred, average='macro')
    recall = recall_score(labels_oh, pred, average='macro')
    print(confusion_matrix(gt_labels, idx))

    return macro_f1_score, precision, recall


def compute_cohort_metrics(test_pred, gt_labels, cohort):

    test_cohorts = np.array(['NKI', 'TRANSBIG', 'UNT', 'UPP', 'VDX'])
    all_metrics = pd.DataFrame(columns=('cohort', 'accuracy', 'balanced accuracy', 'sensitivity', 'specificity'))

    pred = np.argmax(test_pred, axis=1)
    print("All cohorts metric:")
    all_bacc = balanced_accuracy_score(gt_labels, pred)
    all_acc = accuracy_score(gt_labels, pred)
    all_spec = precision_score(gt_labels, pred, average='macro')
    all_sens = recall_score(gt_labels, pred, average='macro')

    all_metrics = all_metrics.append(pd.DataFrame({'cohort': ['all'], 'accuracy': [all_acc],
                                                   'balanced accuracy': [all_bacc], 'sensitivity': [all_sens],
                                                   'specificity': [all_spec]}), ignore_index=True)

    for i in range(test_cohorts.shape[0]):
        cohort_loc = np.where(cohort == test_cohorts[i])[0]
        preds = pred[cohort_loc]
        labels = gt_labels[cohort_loc]

        bacc = balanced_accuracy_score(labels, preds)
        acc = accuracy_score(labels, preds)
        specificity = precision_score(labels, preds, average='macro')
        sensitivity = recall_score(labels, preds, average='macro')

        all_metrics = all_metrics.append(pd.DataFrame({'cohort': [test_cohorts[i]], 'accuracy': [acc],
                                                       'balanced accuracy': [bacc], 'sensitivity': [sensitivity],
                                                       'specificity': [specificity]}), ignore_index=True)

    return all_metrics



################
# Survival Utils
################
def CoxLoss(survtime, censor, hazard_pred):
    # This calculation credit to Travers Ching https://github.com/traversc/cox-nnet
    # Cox-nnet: An artificial neural network method for prognosis prediction of high-throughput omics data
    current_batch_len = len(survtime)
    R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
    for i in range(current_batch_len):
        for j in range(current_batch_len):
            R_mat[i, j] = survtime[j] >= survtime[i]

    R_mat = torch.FloatTensor(R_mat).cuda()
    theta = hazard_pred.reshape(-1)
    exp_theta = torch.exp(theta)
    # print("censor and theta shape:", censor.shape, theta.shape)
    loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta*R_mat, dim=1))) * censor)
    return loss_cox



def accuracy_cox(hazardsdata, labels):
    # This accuracy is based on estimated survival events against true survival events
    median = np.median(hazardsdata)
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazards_dichotomize[hazardsdata > median] = 1
    correct = np.sum(hazards_dichotomize == labels)
    return correct / len(labels)


def cox_log_rank(hazardsdata, labels, survtime_all):
    median = np.median(hazardsdata)
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazards_dichotomize[hazardsdata > median] = 1
    idx = hazards_dichotomize == 0
    T1 = survtime_all[idx]
    T2 = survtime_all[~idx]
    E1 = labels[idx]
    E2 = labels[~idx]
    results = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
    pvalue_pred = results.p_value
    return(pvalue_pred)


def CIndex(hazards, labels, survtime_all):
    concord = 0.
    total = 0.
    N_test = labels.shape[0]
    for i in range(N_test):
        if labels[i] == 1:
            for j in range(N_test):
                if survtime_all[j] > survtime_all[i]:
                    total += 1
                    if hazards[j] < hazards[i]: concord += 1
                    elif hazards[j] < hazards[i]: concord += 0.5

    return(concord/total)


def CIndex_lifeline(hazards, labels, survtime_all):
    return(concordance_index(survtime_all, -hazards, labels))



################
# Layer Utils
################
def define_act_layer(act_type='Tanh'):
    if act_type == 'Tanh':
        act_layer = nn.Tanh()
    elif act_type == 'ReLU':
        act_layer = nn.ReLU()
    elif act_type == 'Sigmoid':
        act_layer = nn.Sigmoid()
    elif act_type == 'LSM':
        act_layer = nn.LogSoftmax(dim=1)
    elif act_type == "none":
        act_layer = None
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act_type)
    return act_layer
