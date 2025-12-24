from microbatch_engine.config import DEVICE, MICROBATCHS, LR, BATCH_SIZE, EPOCHS
from microbatch_engine.models import DeepCNN
from microbatch_engine.data import SyntheticData, generate_quadrant_batch
from microbatch_engine.engine import MicroBatchEngine
from microbatch_engine.utils import seed_all

from torch import nn, optim
from torch.utils.data import DataLoader

seed_all(42)

def oom_demo():
    ### How to run this demo
    ### Set BATCH_SIZE = 512 to force OOM with MICROBATCHS=1 in config.py
    ### Adjusting MICROBATCHS to 8 to make training pass
    ### Make sure img_size is set to 224

    highres_img, highre_y = generate_quadrant_batch(BATCH_SIZE, img_size=224)
    highres_img = highres_img.to(DEVICE)
    highre_y = highre_y.to(DEVICE)

    ds = SyntheticData(highres_img, highre_y)
    data_loader = DataLoader(ds,batch_size=BATCH_SIZE,shuffle=True, drop_last=True)

    # Set up model, Enginee, loss function and optimizer
    model = DeepCNN()
    model = model.to(DEVICE)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    engine = MicroBatchEngine(model, optimizer, loss_fn, MICROBATCHS)

    epoch_loss = 0
    epoch_samples = 0

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        epoch_samples = 0

        for x,y in data_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            
            metrics = engine.train_step((x, y))

            B = len(y)
            epoch_loss += metrics["loss"] * B
            epoch_samples += B

        print(f"Epoch: {epoch} | Avg Loss per sample: {epoch_loss/epoch_samples}")