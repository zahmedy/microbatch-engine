import torch
import torch.nn as nn

from microbatch_engine.config import MAXPOOL_KERNEL, STRIDE, PADDING, KERNEL_SIZE, CLASSES

class SimpleCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, KERNEL_SIZE ,stride=STRIDE, padding=PADDING),
            nn.ReLU(),
            nn.MaxPool2d(MAXPOOL_KERNEL),
            nn.Conv2d(8, 16, KERNEL_SIZE,stride=STRIDE, padding=PADDING),
            nn.ReLU(),
            nn.MaxPool2d(MAXPOOL_KERNEL)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16*7*7, 128),
            nn.ReLU(),
            nn.Linear(128 ,CLASSES)
        )

    def forward(self, x):
        x = self.features(x)
        print(x.shape)
        logits = self.classifier(x)
        print(x.shape)

        return logits 


if __name__ == "__main__":
    random_input = torch.rand([1, 1, 28, 28])
    print(random_input.shape)
    model = SimpleCNN()
    logits = model(random_input)
    print(logits)
    