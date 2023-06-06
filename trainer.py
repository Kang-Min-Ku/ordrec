import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torchmetrics import RetrievalNormalizedDCG
from torchmetrics.functional import retrieval_normalized_dcg
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
                 loss_func=torch.nn.BCELoss,
                 valid_dict=None,
                 test_dict=None,):
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
        self.valid_dict = valid_dict
        self.test_dict = test_dict

        self.eval_func = self.eval_implicit_matrix
        if params["use_dict"]:
            self.eval_func = self.eval_implicit

        self.top_k = 50
        self.ndcg = RetrievalNormalizedDCG(k=self.top_k)
        
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
            epoch_loss = 0
            for batch_idx in range(n_batch):
                self.optimizer.zero_grad()
                batch_users = user_idx[batch_idx * self.batch_size: min(self.num_users, (batch_idx+1)*self.batch_size)]
                rating = self.model.train_batch(batch_users, self.train_adj)
                loss = self.loss_func(self.activation(rating), self.train_target[batch_users].to_dense())
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss
            if epoch % 1 == 0:
                print("--------------------------------------------------")
                prec, recall, ndcg = self.eval_func(self.valid_adj)
                print(f"[AE] epoch: {epoch}, loss: {epoch_loss}")
                print(f"(AE VALID) prec@{self.top_k} {prec}, recall@{self.top_k} {recall}, ndcg@{self.top_k} {ndcg}")
                if self.valid_dict != None:
                    prec, recall, ndcg = self.eval_implicit(self.valid_dict)
                    print("raw by raw:",ndcg)
                print("--------------------------------------------------")

    def test(self):
        prec, recall, ndcg = self.eval_func(self.test_adj)
        print("Test Result")
        print(f"(AE VALID) prec@{self.top_k} {prec}, recall@{self.top_k} {recall}, ndcg@{self.top_k} {ndcg}")
    

    def eval_implicit_matrix(self, targets):
        start = time.time()
        user_idx = torch.arange(self.num_users).cuda()
        prec_list = []
        recall_list = []
        ndcg_list = []
        with torch.no_grad():
            # self.model.eval()
            valid_batchsize = self.batch_size
            n_batch = self.num_users // valid_batchsize + 1
            for batch_idx in range(n_batch):
                batch_users = user_idx[batch_idx * valid_batchsize: min(self.num_users, (batch_idx+1)*valid_batchsize)]
                rating = self.model.train_batch(batch_users, self.train_adj)
                rating = self.activation(rating)
                ndcg_idx = torch.tensor([[i] * self.num_items for i in range(len(batch_users))]).cuda()
                ndcg_k = self.ndcg(rating, targets[batch_users].to_dense(), indexes=ndcg_idx)
                ndcg_list.append(ndcg_k)
                
        print("Executed evaluation:",time.time() - start)
        print(ndcg_k)
        
        return 0, 0, torch.mean(torch.tensor(ndcg_list))

    def eval_implicit(self, targets):
        start = time.time()
        user_idx = torch.arange(self.num_users).cuda()
        prec_list = []
        recall_list = []
        ndcg_list = []
        with torch.no_grad():
            # self.model.eval()
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