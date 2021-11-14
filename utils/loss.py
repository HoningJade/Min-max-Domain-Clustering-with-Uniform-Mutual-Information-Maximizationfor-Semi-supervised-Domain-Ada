import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Function


class GradReverse(Function):
    def __init__(self, lambd):
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
    out_t1 = F.softmax(out_t1)
    loss_adent = lamda * torch.mean(torch.sum(out_t1 *
                                              (torch.log(out_t1 + 1e-5)), 1))
    return loss_adent

def adaac(F1, f_Q, f_K, feat, lamda, feat1=None, eta=1.0):
    if not feat1:
        feat1 = feat
    prob = F.softmax(F1(feat, reverse=True, eta=eta), dim=-1)
    prob1 = F.softmax(F1(feat1, reverse=True, eta=eta), dim=-1)
    Q = f_Q(feat)
    K = f_K(feat)
    simi = torch.sigmoid(Q.matmul(K.t()) / np.sqrt(Q.shape[1]))
    P = prob.matmul(prob1.t())
    loss_adaac = lamda * torch.mean((simi * torch.log(P + 1e-5) + (1 - simi) * torch.log(1 - P + 1e-5)))
    return loss_adaac
