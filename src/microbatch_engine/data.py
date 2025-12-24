
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

from microbatch_engine.config import BATCH_SIZE, MEAN, STD

class SyntheticData(Dataset):
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

def get_dataloaders():

    transformer = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    )

    train_dataset = datasets.CIFAR10("data/", 
                                    train=True,
                                    download=True,
                                    transform=transformer,)
    
    test_dataset = datasets.CIFAR10("data/", 
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


if __name__ == "__main__":
    train_dl, test_dl = get_dataloaders()

    x, y = next(iter(train_dl))

    print(x.shape)
    print(y.shape)
    print(x.dtype)
    print(y.dtype)