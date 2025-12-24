from torch import nn

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
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            loss += scaled_loss.item()
            microbatch_losses.append(loss_i.item())
        
        self.optimizer.step()
        
        return {"loss": loss, "microbatch_losses": microbatch_losses}
            

if __name__ == "__main__":
    pass
