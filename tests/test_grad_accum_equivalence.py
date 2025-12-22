from microbatch_engine.models import SimpleCNN
from microbatch_engine.data import get_dataloaders
from microbatch_engine.config import DEVICE, MICROBATCHS
from microbatch_engine.utils import seed_all

from torch import nn


def test_gradient_accum_equivalence():

    seed_all(42)

    model_full = SimpleCNN()
    model_micro = SimpleCNN()

    model_full_stat = model_full.state_dict()
    model_micro.load_state_dict(model_full_stat)

    model_full.to(DEVICE)
    model_micro.to(DEVICE)

    # Use test loader as shuffle set to False
    _, test_loader = get_dataloaders()
    # single_batch 
    x, y = next(iter(test_loader))
    x, y = x.to(DEVICE), y.to(DEVICE)

    loss_fn = nn.CrossEntropyLoss()

    ### Full Model Run
    model_full.zero_grad()
    full_logits = model_full(x)
    loss = loss_fn(full_logits, y)
    loss.backward()

    grad_dict = {name: param.grad.detach().clone() for name, param in model_full.named_parameters() if param.grad is not None}
    # save_path = "tests/test_states/full_gradients.pt"
    # torch.save(grad_dict, save_path)

    ### Microbatcch model run
    model_micro.zero_grad()
    x_chunk = x.chunk(MICROBATCHS, dim=0)
    y_chunk = y.chunk(MICROBATCHS, dim=0)

    micro_loss = 0
    for x_i, y_i in zip(x_chunk, y_chunk):
        micro_logits = model_micro(x_i)
        loss_i = loss_fn(micro_logits, y_i)
        loss_scaled = loss_i * (x_i.shape[0]/x.shape[0])
        micro_loss += loss_scaled
        loss_scaled.backward()

    micro_grad_dict = {name: param.grad.detach().clone() for name, param in model_micro.named_parameters() if param.grad is not None}
    # save_path = "tests/test_states/micro_gradients.pt"
    # torch.save(micro_grad_dict, save_path)

    tolrance = 1e-5

    # Load gradients 
    # full_grads = torch.load("tests/test_states/full_gradients.pt")
    # micro_grads = torch.load("tests/test_states/micro_gradients.pt")

    for name, _ in model_full.named_parameters():
        max_diff = (grad_dict[name] - micro_grad_dict[name]).abs().max()
        
        assert max_diff.item() < tolrance, f"{name}: max_diff={max_diff}"

    





