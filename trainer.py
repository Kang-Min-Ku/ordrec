import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor
import torch.optim as optim
import numpy as np
from utils.util import compute_metrics
import time
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
        self.valid_adj = valid_adj
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
        self.valid_batch = True
        

    def train(self):
        n_batch = self.num_users// self.batch_size + 1
        
        for epoch in range(self.epochs):
            user_idx = torch.randperm(self.num_users).cuda()
            for batch_idx in range(n_batch):
                batch_users = user_idx[batch_idx * self.batch_size: min(self.num_users, (batch_idx+1)*self.batch_size)]
                rating = self.model.train_batch(batch_users, self.train_adj)
                                
                loss = self.loss_func(self.activation(rating), self.train_target[batch_users].to_dense())
                # print(f"Epoch:{epoch} Loss:{loss}")
                loss.backward()
                self.optimizer.step()
                # print(self.model.x(batch_users[:5]))
                # print(f"[AE] epoch: {epoch}, loss: {loss}")
            print("--------------------------------------------------")
            prec, recall, ndcg = self.eval_implicit(self.valid_adj, user_idx)
            print(f"[AE] epoch: {epoch}, loss: {loss}")
            print(f"(AE VALID) prec@{self.top_k} {prec:.5f}, recall@{self.top_k} {recall:.5f}, ndcg@{self.top_k} {ndcg:.5f}")

    def test(self):
        # pred = self.activation(self.model.rating())
        user_idx = torch.randperm(self.num_users).cuda()
        prec, recall, ndcg = self.eval_implicit(self.test_adj, user_idx)
        print("Test Result")
        print(f"(AE VALID) prec@{self.top_k} {prec}, recall@{self.top_k} {recall}, ndcg@{self.top_k} {ndcg}")
    


    def eval_implicit(self, targets, user_idx):
        start = time.time()
        prec_list = []
        recall_list = []
        ndcg_list = []
        with torch.no_grad():
            valid_batchsize = 1024
            n_batch = self.num_users // valid_batchsize + 1
            for batch_idx in range(n_batch):
                batch_users = user_idx[batch_idx * valid_batchsize: min(self.num_users, (batch_idx+1)*valid_batchsize)]
                rating = self.model.rating(batch_users)
                rating = self.activation(rating)
                # print(rating[:5])
                pred = torch.where(self.train_target[batch_users].to_dense() > 0.5, torch.tensor(0,dtype=torch.float).cuda(), rating)
                pred = pred.detach().cpu().numpy()
                # print(pred[:5])
                pred = np.argsort(pred, axis=1)[::-1]
                # print(pred[:5])
                for i, user_id in enumerate(batch_users.detach().cpu().numpy()):
                    target = targets[user_id]
                    # print(user_id)
                    prec_k, recall_k, ndcg_k = compute_metrics(pred[i], target, self.top_k)
                    prec_list.append(prec_k)
                    recall_list.append(recall_k)
                    ndcg_list.append(ndcg_k)
        
        print("Executed evaluation:",time.time() - start)
        
        return np.mean(prec_list), np.mean(recall_list), np.mean(ndcg_list)