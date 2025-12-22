
import torch

KERNEL_SIZE = 3
MAXPOOL_KERNEL = 2
STRIDE = 1
PADDING = 1
CLASSES = 10
BATCH_SIZE = 64
MEAN = 0.1307
STD = 0.3081
LR = 0.001
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
EPOCHS = 5
MICROBATCHS = 4