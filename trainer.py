import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor
import torch.optim as optim
import numpy as np
from utils.util import compute_metrics

from model.ordrec import OrdRec
from model.GONN import GONN

class Trainer:
    def __init__(self,
                 params,
                 num_users,
                 num_items,
                 train_adj=None,
                 valid_adj=None,
                 test_adj=None,
                 optimizer=optim.Adam,
                 loss_func=torch.nn.BCELoss):
        self.params = params
        self.device = "cuda" if params["use_cuda"] and torch.cuda.is_available() else "cpu"
        print(self.device)
        self.num_users = num_users
        self.num_items = num_items

        assert train_adj is not None, "Require train adj"
        assert test_adj is not None, "Require test adj"
        self.train_adj = train_adj.cuda()
        self.valid_adj = valid_adj.cuda()
        self.test_adj = test_adj
        
        self.top_k = 20
        
        self.train_target = train_adj[:self.num_users, self.num_users:].cuda()

        self.model = GONN(params, self.num_users, self.num_items)
        self.model = self.model.to(self.device)

        self.optimizer = optimizer(self.model.parameters(), lr=params["learning_rate"])
        self.loss_func = loss_func()
        
        self.epochs = params["epochs"]
        self.batch_size = params["batch_size"]
        self.activation = torch.nn.Sigmoid()
        

    def train(self):
        n_batch = self.num_users// self.batch_size + 1
        
        for epoch in range(self.epochs):
            user_idx = torch.randperm(self.num_users).cuda()
            for batch_idx in range(1):
                batch_users = user_idx[batch_idx * self.batch_size: min(self.num_users, (batch_idx+1)*self.batch_size)]
                rating = self.model.train_batch(batch_users, self.train_adj)
                                
                loss = self.loss_func(self.activation(rating), self.train_target[batch_users].to_dense())
                # print(f"Epoch:{epoch} Loss:{loss}")
                loss.backward()
                self.optimizer.step()
                # print(self.model.x(batch_users[:5]))
            
            print("--------------------------------------------------")
            prec, recall, ndcg = self.eval_implicit(self.top_k)
            print("[AE] epoch %d, loss: %f"%(epoch))
            print(f"(AE VALID) prec@{self.eval_topk} {prec:.5f}, recall@{self.eval_topk} {recall:.5f}, ndcg@{self.eval_topk} {ndcg:.5f}")

    def test(self):
        # pred = self.activation(self.model.rating())
        prec, recall, ndcg = self.eval_implicit(self.top_k)
        print("Test Result")
        print(f"(AE VALID) prec@{self.eval_topk} {prec:.5f}, recall@{self.eval_topk} {recall:.5f}, ndcg@{self.eval_topk} {ndcg:.5f}")
    
    def eval_implicit(self, top_k):        
        with torch.no_grad():
            pred = self.model.rating().to("cpu")
            # pred = self.activation()
            print(pred[:5])
            train_user_id = self.train_adj.storage.row()
            train_item_id = self.train_adj.storage.col()
            pred[train_user_id, train_item_id] = 0
            pred = torch.argsort(pred)[::-1]
            print(pred[:5])
            target = torch.where(self.valid_adj >= 0.5)[0]
            prec_k, recall_k, ndcg_k = compute_metrics(pred, target, top_k)


        return prec_k, recall_k, ndcg_k
