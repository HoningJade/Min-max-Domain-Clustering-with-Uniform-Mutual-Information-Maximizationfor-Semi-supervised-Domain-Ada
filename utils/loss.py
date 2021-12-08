from torch.autograd.variable import Variable
import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Function
import torch.nn as nn


class GradReverse(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * -self.lambd)


def grad_reverse(x, lambd=1.0):
    return GradReverse(lambd)(x)


def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) /
                    (1.0 + np.exp(-alpha * iter_num / max_iter)) -
                    (high - low) + low)


def entropy(F1, feat, lamda, eta=1.0):
    out_t1 = F1(feat, reverse=True, eta=-eta)
    out_t1 = F.softmax(out_t1)
    loss_ent = -lamda * torch.mean(torch.sum(out_t1 *
                                             (torch.log(out_t1 + 1e-5)), 1))
    return loss_ent


def adentropy(F1, feat, lamda, eta=1.0):
    out_t1 = F1(feat, reverse=True, eta=eta)
    out_t1 = F.softmax(out_t1, dim=-1)  # TODO: double check dim=-1 is correct
    loss_adent = lamda * torch.mean(torch.sum(out_t1 *
                                              (torch.log(out_t1 + 1e-5)), 1))
    return loss_adent

def JSDloss(p, q, softmax=False):
    if softmax:
        p = F.softmax(p, dim=1)
        q = F.softmax(q, dim=1)

    m = Variable((p + q) / 2)
    jsd = (F.kl_div(p, m) + F.kl_div(q, m))
    return jsd


def MomentumJSDLoss(source_distribution, target_distribution, source_out, target_out_all, m=0.99):
    source_out = F.softmax(source_out, dim=1)
    target_out_all = F.softmax(target_out_all, dim=1)
    src_dist = torch.Tensor(source_distribution).cuda()
    tar_dsit = torch.Tensor(target_distribution).cuda()

    src_dist = m * src_dist + (1 - m)*torch.mean(source_out, dim=0)
    tar_dsit = m * tar_dsit + (1 - m)*torch.mean(target_out_all, dim=0)

    return JSDloss(src_dist[None, :], tar_dsit[None, :]), src_dist.cpu().detach().numpy(), tar_dsit.cpu().detach().numpy()
