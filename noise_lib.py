import abc
import torch
import torch.nn as nn
import numpy as np

def get_noise(config):
    """根据配置选择噪声类型"""
    if config.noise.type == "geometric":
        return GeometricNoise(config.noise.sigma_min, config.noise.sigma_max)
    elif config.noise.type == "loglinear":
        return LogLinearNoise()
    else:
        raise ValueError(f"{config.noise.type} is not a valid noise")

class Noise(abc.ABC, nn.Module):
    """噪声基类"""
    def forward(self, t):
        return self.total_noise(t), self.rate_noise(t)

    @abc.abstractmethod
    def rate_noise(self, t):
        """噪声变化率"""
        pass

    @abc.abstractmethod
    def total_noise(self, t):
        """总噪声"""
        pass

class GeometricNoise(Noise):
    """几何噪声"""
    def __init__(self, sigma_min=1e-3, sigma_max=1, learnable=False):
        super().__init__()
        self.sigmas = 1.0 * torch.tensor([sigma_min, sigma_max])
        if learnable:
            self.sigmas = nn.Parameter(self.sigmas)
        self.empty = nn.Parameter(torch.tensor(0.0))

    def rate_noise(self, t):
        return self.sigmas[0] ** (1 - t) * self.sigmas[1] ** t * (self.sigmas[1].log() - self.sigmas[0].log())

    def total_noise(self, t):
        return self.sigmas[0] ** (1 - t) * self.sigmas[1] ** t

class LogLinearNoise(Noise):
    """对数线性噪声"""
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps
        self.empty = nn.Parameter(torch.tensor(0.0))

    def rate_noise(self, t):
        return (1 - self.eps) / (1 - (1 - self.eps) * t)

    def total_noise(self, t):
        return -torch.log1p(-(1 - self.eps) * t)