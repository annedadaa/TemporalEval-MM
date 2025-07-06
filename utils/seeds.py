import random
import numpy as np
import torch
from transformers import set_seed

def set_all_seeds(seed: int) -> None:
    """
    Set all random seeds for reproducibility.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)
