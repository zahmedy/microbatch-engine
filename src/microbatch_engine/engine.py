import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from microbatch_engine.config import DEVICE, MICROBATCHS, LR, BATCH_SIZE, EPOCHS
from microbatch_engine.models import DeepCNN
from microbatch_engine.data import SyntheticData



class MicroBatchEngine():
    def __init__(self, model, optimizer, loss_fn, microbatches=4) -> None:
        self.model = model.to()
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.microbatches = microbatches

    def train_step(self, batch):
        x, y = batch            #((batch, channels, hieght, width), (batch,))
        assert x.dim() == 4
        assert y.dim() == 1

        self.optimizer.zero_grad()

        # Split to chunks equal to microbatchs along batch dim
        x_chunks = x.chunk(self.microbatches, dim=0)
        y_chunks = y.chunk(self.microbatches, dim=0)

        B = x.shape[0] # Full batch size from tensor
        assert B % self.microbatches == 0

        microbatch_losses = []
        loss = 0
        for microbatch in range(self.microbatches):
            # Work on first microbatch
            x_i = x_chunks[microbatch]
            y_i = y_chunks[microbatch]

            logits = self.model(x_i)

            loss_i = self.loss_fn(logits, y_i)
            b = x_i.shape[0]
            scale = b / B
            scaled_loss = loss_i * scale
            scaled_loss.backward()
            loss += scaled_loss.item()
            microbatch_losses.append(loss_i.item())
        
        self.optimizer.step()
        
        return {"loss": loss, "microbatch_losses": microbatch_losses}
            

if __name__ == "__main__":
    ### STRESS RUN TO FORCE OOM 
    highres_img = torch.randn([BATCH_SIZE, 3, 224, 224])
    highre_y = torch.randint(0, 10, (BATCH_SIZE,))
    highres_img = highres_img.to(DEVICE)
    highre_y = highre_y.to(DEVICE)

    ds = SyntheticData(highres_img, highre_y)
    data_loader = DataLoader(ds,batch_size=BATCH_SIZE,shuffle=True, drop_last=True)

    # Set up model, Enginee, loss function and optimizer
    model = DeepCNN()
    model = model.to(DEVICE)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
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
