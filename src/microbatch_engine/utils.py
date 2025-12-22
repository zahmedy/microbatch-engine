from microbatch_engine.models import SimpleCNN
from microbatch_engine.engine import MicroBatchEngine
from microbatch_engine.data import get_dataloaders
from microbatch_engine.config import DEVICE, LR, EPOCHS

from torch import nn, optim
import random
import numpy as np
import torch

def train_one_epoch(dataloader):
    model = SimpleCNN()
    model.to(DEVICE)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    total_loss = 0
    for images, labels in dataloader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        logits = model(images)
        loss = loss_fn(logits, labels)
        total_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return total_loss, model.get_parameter

def seed_all(seed):
    """Set all randomness to seed for reproducaiblity"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.mps.manual_seed(seed)

        


    


