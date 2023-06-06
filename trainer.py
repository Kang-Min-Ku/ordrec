import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor
import torch.optim as optim
import numpy as np
from utils.util import compute_metrics
from utils.train import check_improvement
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
        #early stopping
        self.best_score = None
        self.early_stop_policy = params["early_stop_policy"]
        self.patience = 0
        self.early_stop = False

    def train(self):
        self.patience = 0
        n_batch = self.num_users// self.batch_size + 1
        
        for epoch in range(self.epochs):
            user_idx = torch.randperm(self.num_users).cuda()
            epoch_loss = 0
            for batch_idx in range(n_batch):
                self.optimizer.zero_grad()
                batch_users = user_idx[batch_idx * self.batch_size: min(self.num_users, (batch_idx+1)*self.batch_size)]
                #print(self.train_adj[4].to_dense().sum())
                rating = self.model.train_batch(batch_users, self.train_adj)
                loss = self.loss_func(self.activation(rating), self.train_target[batch_users].to_dense())
                #print((self.train_adj[4,self.num_users:].to_dense() == self.train_target[4].to_dense()).sum())
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss
            if epoch % self.params["print_every"] == 0:
                print("--------------------------------------------------")
                prec, recall, ndcg = self.eval_implicit(self.valid_adj)
                print(f"[AE] epoch: {epoch}, loss: {epoch_loss}")
                print(f"(AE VALID) prec@{self.top_k} {prec}, recall@{self.top_k} {recall}, ndcg@{self.top_k} {ndcg}")
                print("--------------------------------------------------")
                # early stop
                if self.early_stop_policy in ["loss"]:
                    self.patience, self.best_score = check_improvement(self.patience, self.best_score, epoch_loss, mode="min")
                elif self.early_stop_policy in ["prec", "recall", "ndcg"]:
                    self.patience, self.best_score = check_improvement(self.patience, self.best_score, epoch_loss, mode="max")

                if self.patience > self.params["early_stop_threshold"]:
                    self.early_stop = True

            if self.params["do_early_stop"] and self.early_stop:
                print("==================================================")
                print(f"Early Stop Training at epoch {epoch}")
                print("==================================================")
                break

    def test(self):
        prec, recall, ndcg = self.eval_implicit(self.test_adj)
        print("Test Result")
        print(f"(AE VALID) prec@{self.top_k} {prec}, recall@{self.top_k} {recall}, ndcg@{self.top_k} {ndcg}")
    


    def eval_implicit(self, targets):
        start = time.time()
        user_idx = torch.arange(self.num_users).cuda()
        prec_list = []
        recall_list = []
        ndcg_list = []
        with torch.no_grad():
            self.model.eval()
            valid_batchsize = self.batch_size
            n_batch = self.num_users // valid_batchsize + 1
            for batch_idx in range(n_batch):
                batch_users = user_idx[batch_idx * valid_batchsize: min(self.num_users, (batch_idx+1)*valid_batchsize)]
                rating = self.model.train_batch(batch_users, self.train_adj)
                rating = self.activation(rating)
                pred = torch.where(self.train_target[batch_users].to_dense() > 0, torch.tensor(-2.0,dtype=torch.float).cuda(), rating)
                pred = pred.detach().cpu().numpy()
                pred_u = np.argsort(pred, axis=1)[:,::-1]

                for i, user_id in enumerate(batch_users.detach().cpu().numpy()):
                    target = targets[user_id]
                    prec_k, recall_k, ndcg_k = compute_metrics(pred_u[i], target, self.top_k)
                    prec_list.append(prec_k)
                    recall_list.append(recall_k)
                    ndcg_list.append(ndcg_k)
        
        print("Executed evaluation:",time.time() - start)
        
        return np.mean(prec_list), np.mean(recall_list), np.mean(ndcg_list)