from microbatch_engine.models import SimpleCNN
from microbatch_engine.data import get_dataloaders
from microbatch_engine.engine import MicroBatchEngine
from microbatch_engine.config import LR, DEVICE

from torch import optim, nn
import torch

def train(model, engine, loss_fn, optimizer):
    epoch_loss = 0
    epoch_samples = 0

    for epoch in range(5):
        model.train()
        epoch_loss = 0
        epoch_samples = 0

        for x,y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            
            metrics = engine.train_step((x, y))

            B = len(y)
            epoch_loss += metrics["loss"] * B
            epoch_samples += B

        print(f"Epoch: {epoch} | Avg Loss per sample: {epoch_loss/epoch_samples}")

def evaluation():
    model.eval()
    test_loss_sum = 0
    test_correct = 0
    test_samples = 0

    for x, y in test_loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        with torch.no_grad():
            logits = model(x)
            loss = loss_fn(logits, y)
            B = len(y)
            test_loss_sum += loss.item() * B
            pred = logits.argmax(dim=1)
            test_correct += (pred == y).sum().item()
            test_samples += B
    
    avg_test_loss = test_loss_sum/test_samples
    test_acc = test_correct/test_samples
    print(f"Avg Test Loss: {avg_test_loss} | Test Accuracy: {test_acc} | Accuracy %: {test_acc * 100}")


if __name__ == "__main__":
    ## Get data to train 
    train_loader, test_loader = get_dataloaders()

    # Set up model, Enginee, loss function and optimizer
    model = SimpleCNN()
    model = model.to(DEVICE)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    engine = MicroBatchEngine(model, optimizer, loss_fn)
    train(model, engine, loss_fn, optimizer)
    evaluation()