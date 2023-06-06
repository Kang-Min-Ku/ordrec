import numpy as np
import torch

def check_improvement(count, best, current, mode="max"):
    if best is None:
        best = current
    elif mode == "max":
        if current >= best:
            best = current
            count = 0
        else:
            count += 1
    elif mode == "min":
        if current <= best:
            best = current
            count = 0
        else:
            count += 1

    return count, best
