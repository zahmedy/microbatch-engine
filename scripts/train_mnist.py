from microbatch_engine.models import SimpleCNN
from microbatch_engine.data import get_dataloaders
from microbatch_engine.engine import MicroBatchEngine
from microbatch_engine.config import LR, DEVICE

from torch import optim, nn

## Get data to train 
train_loader, test_loader = get_dataloaders()

# Set up model, Enginee, loss function and optimizer
model = SimpleCNN()
model = model.to(DEVICE)

loss_fn = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=LR)

engine = MicroBatchEngine(model, optimizer, loss_fn)

(x, y) = next(iter(train_loader))

(x, y) = (x.to(DEVICE), y.to(DEVICE))

metrics = engine.train_step((x, y))

print(metrics["loss"])