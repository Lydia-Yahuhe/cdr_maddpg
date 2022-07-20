import torch as th
import torch.nn.functional as F

import numpy as np
from torchviz import make_dot

BoolTensor = th.BoolTensor
FloatTensor = th.FloatTensor
root = 'trained/model/'


def net_visual(dim_input, net, name):
    xs = [th.randn(*dim).requires_grad_(True) for dim in dim_input]  # 定义一个网络的输入值
    y = net(*xs)  # 获取网络的预测值
    net_vis = make_dot(y, params=dict(list(net.named_parameters()) + [('x', x) for x in xs]))
    net_vis.format = "png"
    # 指定文件生成的文件夹
    net_vis.directory = "trained/data/{}".format(name)
    # 生成文件
    net_vis.view()


def soft_update(target, source, t):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - t) * target_param.data + t * source_param.data)


def hard_update(target, source):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(source_param.data)


def average_gradients(target, size):
    """ Gradient averaging. """
    for param in target.parameters():
        param.grad.data /= size


def gumbel_softmax(logits, discrete_list, noisy=False, var=1.0):
    actions = []
    for action in th.split(logits, discrete_list, dim=-1):
        if noisy:
            act_noisy = th.randn(action.shape) * var
            action += act_noisy
        actions.append(F.gumbel_softmax(action, hard=True))
    return th.cat(actions, dim=-1)




