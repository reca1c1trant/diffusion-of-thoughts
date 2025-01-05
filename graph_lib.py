import abc
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import custom_fwd, custom_bwd

def get_graph(config, device):
    """根据配置选择图类型"""
    if config.graph.type == "uniform":
        return Uniform(config.tokens)
    elif config.graph.type == "absorb":
        return Absorbing(config.tokens)
    else:
        raise ValueError(f"Graph {config.graph.type} not valid")

class Graph(abc.ABC):
    """图基类"""
    @property
    @abc.abstractmethod
    def dim(self):
        pass

    @property
    def absorb(self):
        """是否为吸收态"""
        pass

    @abc.abstractmethod
    def rate(self, i):
        """计算速率矩阵Q的第i列"""
        pass

    @abc.abstractmethod
    def transp_rate(self, i):
        """计算速率矩阵Q的第i行"""
        pass

    @abc.abstractmethod
    def transition(self, i, sigma):
        """计算转移矩阵的第i列"""
        pass

    def sample_transition(self, i, sigma):
        """采样转移向量"""
        transition_vector = self.transition(i, sigma)
        return torch.multinomial(transition_vector, 1).squeeze(-1)

class Uniform(Graph):
    """均匀图"""
    def __init__(self, dim):
        self._dim = dim

    @property
    def dim(self):
        return self._dim
    
    @property
    def absorb(self):
        return False

    def rate(self, i):
        edge = torch.ones(*i.shape, self.dim, device=i.device) / self.dim
        edge = edge.scatter(-1, i[..., None], - (self.dim - 1) / self.dim)
        return edge

    def transp_rate(self, i):
        return self.rate(i)

    def transition(self, i, sigma):
        trans = torch.ones(*i.shape, self.dim, device=i.device) * (1 - (-sigma[..., None]).exp()) / self.dim
        trans = trans.scatter(-1, i[..., None], torch.zeros_like(trans))
        trans = trans.scatter(-1, i[..., None], 1 - trans.sum(dim=-1, keepdim=True))
        return trans

class Absorbing(Graph):
    """吸收态图"""
    def __init__(self, dim):
        super().__init__()
        self._dim = dim

    @property
    def dim(self):
        return self._dim + 1
    
    @property
    def absorb(self):
        return True

    def rate(self, i):
        return F.one_hot((self.dim - 1) * torch.ones_like(i), num_classes=self.dim) - F.one_hot(i, num_classes=self.dim)

    def transp_rate(self, i):
        edge = -F.one_hot(i, num_classes=self.dim)
        edge[i == self.dim - 1] += 1
        return edge

    def transition(self, i, sigma):
        edge = (-sigma).exp() * F.one_hot(i, num_classes=self.dim)
        edge += torch.where(
            i == self.dim - 1,
            1 - (-sigma).squeeze(-1).exp(),
            0
        )[..., None]
        return edge

    def sample_transition(self, i, sigma):
        move_chance = 1 - (-sigma).exp()
        move_indices = torch.rand(*i.shape, device=i.device) < move_chance
        i_pert = torch.where(move_indices, self.dim - 1, i)
        return i_pert