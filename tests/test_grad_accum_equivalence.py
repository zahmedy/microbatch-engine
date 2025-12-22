import pytest 

from microbatch_engine.models import SimpleCNN
from microbatch_engine.engine import MicroBatchEngine
from microbatch_engine.data import get_dataloaders
from microbatch_engine.config import DEVICE


def test_gradient_equivalence():
    


