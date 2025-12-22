import torch


class MicroBatchEngine():
    def __init__(self, model, optimizer, loss_fn, microbatches=4) -> None:
        self.model = model 
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.microbatches = microbatches

    def train_step(self, batch):
        x, y = batch            #((channels, hieght, width), (labels,))
        assert x.dim() == 4
        assert y.dim() == 1
        assert x[0] % self.microbatches == 0

        self.optimizer.zero_grad()

        B = x.shape[0] # Full batch size from tensor

        microbatch_losses = []
        loss = 0
        for microbatch in range(B/self.microbatches):
            x_i, y_i = batch[microbatch]

            logits = self.model(x_i)
            loss_i = self.loss_fn(logits, y_i)
            b = x_i.shape[0]
            scale = b / B
            scaled_loss = loss_i * scale
            scaled_loss.backward()
            loss += loss_i
            microbatch_losses.append(loss_i)
        
        return loss, microbatch_losses
            



