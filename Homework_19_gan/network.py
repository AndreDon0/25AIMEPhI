import torch.nn as nn
import torch
from random import random

class Generator(nn.Module):
    def __init__(self, image_size, latent_dim=100):
        super().__init__()
        self.latent_dim = latent_dim
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, image_size),
            nn.Sigmoid(),
        )

    def forward(self):
        return self.model(torch.randn(self.latent_dim))

class Discriminator(nn.Module):
    def __init__(self, image_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(image_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)