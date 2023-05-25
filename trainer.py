import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor
import torch.optim as optim

from model.ordrec import OrdRec

class Trainer:
    def __init__(self,
                 params,
                 num_users,
                 num_items,
                 train_adj=None,
                 valid_adj=None,
                 test_adj=None,
                 optimizer=optim.Adam,
                 loss_func=F.binary_cross_entropy_with_logits):
        self.params = params
        self.device = "cuda" if params["use_cuda"] and torch.cuda.is_available() else "cpu"

        self.num_users = num_users
        self.num_items = num_items

        assert train_adj is not None, "Require train adj"
        assert test_adj is not None, "Require test adj"
        self.train_adj = train_adj
        self.test_adj = test_adj
        self.valid_adj = valid_adj

        self.model = OrdRec(params, self.num_users, self.num_items)
        self.model = self.model.to(self.device)

        self.optimizer = optimizer(self.model.parameters(), lr=params["learning_rate"])
        self.loss_func = loss_func

    def train(self):
        NotImplementedError
    
    def test(self):
        NotImplementedError