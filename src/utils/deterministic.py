# Stuff
import os
from diffusers.utils.testing_utils import enable_full_determinism

import torch
import numpy as np
import random


def make_deterministic():

    # Just...to make sure
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    # https://huggingface.co/docs/diffusers/en/using-diffusers/reusing_seeds?hardware=GPU#deterministic-algorithms
    enable_full_determinism()

    # https://huggingface.co/docs/diffusers/en/using-diffusers/reusing_seeds?hardware=GPU#deterministic-algorithms
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
