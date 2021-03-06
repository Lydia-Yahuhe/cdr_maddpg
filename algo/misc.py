import os

import torch as th
import torch.nn.functional as F

import numpy as np
from torchviz import make_dot

device = th.device("cuda") if th.cuda.is_available() else th.device("cpu")

BoolTensor = th.BoolTensor
FloatTensor = th.FloatTensor

root = 'trained/'
logs_path = root + 'logs/'
graph_path = root + 'graph/'
model_path = root + 'model/'
if not os.path.exists(logs_path):
    os.mkdir(logs_path)
if not os.path.exists(graph_path):
    os.mkdir(graph_path)
if not os.path.exists(model_path):
    os.mkdir(model_path)


def to_torch(np_array):
    return th.from_numpy(np_array)


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return th.Tensor(size).uniform_(-v, v)


def weight_init(m):
    if isinstance(m, th.nn.Conv2d) or isinstance(m, th.nn.Linear):
        m.weight.data.fill_(0.)
        m.bias.data.fill_(0.)


def net_visual(dim_input, net, name):
    xs = [th.randn(*dim).requires_grad_(True) for dim in dim_input]  # 定义一个网络的输入值
    y = net(*xs)  # 获取网络的预测值
    net_vis = make_dot(y, params=dict(list(net.named_parameters()) + [('x', x) for x in xs]))
    net_vis.format = "png"
    # 指定文件生成的文件夹
    net_vis.directory = graph_path + "{}".format(name)
    # 生成文件
    net_vis.view()


def soft_update(target, source, t):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - t) * target_param.data + t * source_param.data)


def hard_update(target, source):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(source_param.data)


def interpolate_vars(old_vars, new_vars, epsilon):
    """
    Interpolate between two sequences of variables.
    """
    final_vars = old_vars.copy()
    for var_name, value in old_vars.items():
        mean_var_value = th.mean(th.stack([var_seq[var_name] for var_seq in new_vars]), dim=0)
        final_vars[var_name] = value + (mean_var_value - value) * epsilon
    return final_vars


def set_dynamic_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def gumbel_softmax(logits, discrete_list, noisy=False, var=1.0):
    actions = []
    for action in th.split(logits, discrete_list, dim=-1):
        if noisy:
            act_noisy = th.randn(action.shape) * var
            action += act_noisy
        actions.append(F.gumbel_softmax(action, hard=True))
    return th.cat(actions, dim=-1)
