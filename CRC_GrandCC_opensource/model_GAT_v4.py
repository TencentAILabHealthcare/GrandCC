"""
Compared with the model_GAT_v2, this version tries to interpret the GAT network.
We compute the gradient of y over each node, thus knowing the node importance.
In addition, the GAT layer returns the edge weights, so we know the importance of each edge.
With the important nodes and edges,
we can deduce sub-graphs (pathways) that are important for the disease diagnosis.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGPooling

from torch.optim import Adam
from torch.nn import init, Parameter, LayerNorm
import torch.optim.lr_scheduler as lr_scheduler
from torch_geometric.utils import to_dense_batch, to_dense_adj

from utils import *
from drop_edge_utils.sample import randomedge_sampler
from torch.autograd import Variable
from torch import autograd

class MLA_GNN(torch.nn.Module):
    def __init__(self, opt):
        super(MLA_GNN, self).__init__()
        self.fc_dropout = opt.fc_dropout
        self.GAT_dropout = opt.GAT_dropout
        self.nhids = [5, 8]
        self.nheads = [4, 5]
        self.fc_dim = [128]

        self.conv1 = GATConv(opt.input_dim, self.nhids[0], heads=self.nheads[0],
                             dropout=self.GAT_dropout)

        self.pool1 = torch.nn.Linear(self.nhids[0]*self.nheads[0], 1)

        self.layer_norm0 = LayerNorm(opt.num_nodes)
        self.layer_norm1 = LayerNorm(opt.num_nodes)

        if opt.which_layer == 'cat_all':
            lin_input_dim = 2 * opt.num_nodes
        else:
            lin_input_dim = opt.num_nodes

        self.encoder = nn.Linear(lin_input_dim, self.fc_dim[0])
        self.classifier = nn.Linear(self.fc_dim[0], opt.label_dim)


    def forward(self, x, adj, grad_labels, batch, opt):

        ### layer1
        x = x.requires_grad_()
        x0 = to_dense_batch(torch.mean(x, dim=-1), batch=batch)[0] #[bs, nodes]
        input = x

        ### layer2
        x, (edge_index, att_weights) = self.conv1(x, adj, return_attention_weights=True)
        edge_weights = to_dense_adj(edge_index, batch=batch,
                                    edge_attr=torch.mean(att_weights, 1)) # [bs, nodes, nodes]
        x = F.elu(x)
        x1 = to_dense_batch(self.pool1(x).squeeze(-1), batch=batch)[0] #[bs, nodes]

        if opt.layer_norm == "True":
            x0 = self.layer_norm0(x0)
            x1 = self.layer_norm1(x1)

        if opt.which_layer == 'sum_all':
            x = (x0 + x1)/2
        elif opt.which_layer == 'cat_all':
            x = torch.cat([x0, x1], dim=1)
        elif opt.which_layer == 'layer1':
            x = x0
        elif opt.which_layer == 'layer2':
            x = x1

        GAT_features = x
        features = F.relu(self.encoder(x))
        out = self.classifier(features)

        fc_features = features

        one_hot_labels = torch.zeros(grad_labels.shape[0],
                                     opt.label_dim).cuda().scatter(1, grad_labels.reshape(-1, 1), 1)
        y_c = torch.sum(one_hot_labels*out)

        input.grad = None
        input.retain_grad()
        y_c.backward(retain_graph=True)
        y_c.requres_grad = True
        gradients = torch.maximum(input.grad, torch.zeros_like(input.grad))
        feature_importance = to_dense_batch(torch.mean(gradients, dim=-1), batch=batch)[0]
        # print("feature importance:", feature_importance.shape)

        return GAT_features, fc_features, out, edge_weights, feature_importance


def define_optimizer(opt, model):
    optimizer = None
    if opt.optimizer_type == 'adabound':
        optimizer = adabound.AdaBound(model.parameters(), lr=opt.lr, final_lr=opt.final_lr)
    elif opt.optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), weight_decay=opt.weight_decay)
    elif opt.optimizer_type == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay, initial_accumulator_value=0.1)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % opt.optimizer)
    return optimizer


def define_reg(model):

    for W in model.parameters():
        loss_reg = torch.abs(W).sum()

    return loss_reg


def define_scheduler(opt, optimizer):
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1) / float(opt.num_epochs + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'exp':
        scheduler = lr_scheduler.ExponentialLR(optimizer, 0.1, last_epoch=-1)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

