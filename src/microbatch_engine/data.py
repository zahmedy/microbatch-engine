
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torch

from microbatch_engine.config import BATCH_SIZE, MEAN, STD, MEAN_MNIST, STD_MNIST

class SyntheticData(Dataset):
    """Dataset class to create Synthetic Data"""
    def __init__(self, data, labels) -> None:
        super().__init__()
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sample = self.data[index]
        label = self.labels[index]
        return  sample, label

def get_dataloaders(dataset):
    """Generate Dataloaders from CIFAR10 Dataset"""
    if dataset == "cifar":
        ds = datasets.CIFAR10
    else:
        ds = datasets.FashionMNIST
        MEAN = MEAN_MNIST
        STD = STD_MNIST

    transformer = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    )

    train_dataset = ds("data/", 
                            train=True,
                            download=True,
                            transform=transformer,)
    
    test_dataset = ds("data/", 
                            train=False,
                            download=True,
                            transform=transformer,)

    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=BATCH_SIZE,
                                  drop_last=True,
                                  shuffle=True)
    
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=BATCH_SIZE,
                                 drop_last=True,
                                  shuffle=False)
    
    return train_dataloader, test_dataloader

def generate_quadrant_batch(batch_size, img_size=224, square_size=40):
    images = torch.zeros((batch_size, 3, img_size, img_size))
    labels = torch.zeros(batch_size, dtype=torch.long) # Shape (BATCH_SIZE,)
    
    mid = img_size // 2
    # Adjust margin to ensure the square fits inside the chosen quadrant
    limit = mid - square_size 

    for i in range(batch_size):
        # 1. Randomly choose a quadrant (0, 1, 2, or 3)
        quad = torch.randint(0, 4, (1,)).item()
        labels[i] = quad

        # 2. Set coordinates based on the chosen quadrant
        if quad == 0:   # Top-Left
            y, x = torch.randint(0, limit, (1,)), torch.randint(0, limit, (1,))
        elif quad == 1: # Top-Right
            y, x = torch.randint(0, limit, (1,)), torch.randint(mid, img_size - square_size, (1,))
        elif quad == 2: # Bottom-Left
            y, x = torch.randint(mid, img_size - square_size, (1,)), torch.randint(0, limit, (1,))
        else:           # Bottom-Right
            y, x = torch.randint(mid, img_size - square_size, (1,)), torch.randint(mid, img_size - square_size, (1,))

        # 3. Draw the bright square
        images[i, :, y : y + square_size, x : x + square_size] = 1.0

    print(f"Generated Images shape: {images.shape} Labels shape: {labels.shape}") 
    return images, labels

if __name__ == "__main__":
    train_dl, test_dl = get_dataloaders("cifar")

    x, y = next(iter(train_dl))

    print(x.shape)
    print(y.shape)
    print(x.dtype)
    print(y.dtype)