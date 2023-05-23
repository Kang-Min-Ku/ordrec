import os
import torch
from torch_sparse import SparseTensor

def load_data(path, train_file="train.txt", test_file="test.txt"):
    """
    Make user, item interaction to (# user + # item) x (# user + # item) matrix M.
    Therefore, first item id becomes (# user)

    M[:# user, :# item] = 0
    M[:# user, # item:] = 1 or 0
    M[# user:, :# item] = 1 or 0
    M[# user:, # item:] = 0
    """
    files_in_path = os.listdir(path)
    assert train_file in files_in_path, "No train file"
    assert test_file in files_in_path, "No test file"

    train_file = os.path.join(path, train_file)
    test_file = os.path.join(path, test_file)
    train_unique_users, train_item, train_user = [], [], []
    test_unique_users, test_item, test_user = [], [], []
    n_user = 0
    m_item = 0
    train_data_size = 0
    test_data_size = 0

    with open(train_file, "r") as f:
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip('\n').split(' ')
                items = [int(i) for i in l[1:]]
                uid = int(l[0])
                train_unique_users.append(uid)
                train_user.extend([uid] * len(items))
                train_item.extend(items)
                m_item = max(m_item, max(items))
                n_user = max(n_user, uid)
                train_data_size += len(items)
    train_unique_users = torch.Tensor(train_unique_users).type(torch.long)
    train_user = torch.Tensor(train_user).type(torch.long)
    train_item = torch.Tensor(train_item).type(torch.long)

    with open(test_file) as f:
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip('\n').split(' ')
                items = [int(i) for i in l[1:]]
                uid = int(l[0])
                test_unique_users.append(uid)
                test_user.extend([uid] * len(items))
                test_item.extend(items)
                m_item = max(m_item, max(items))
                n_user = max(n_user, uid)
                test_data_size += len(items)
    m_item += 1
    n_user += 1
    test_unique_users = torch.Tensor(test_unique_users).type(torch.long)
    test_user = torch.Tensor(test_user).type(torch.long)
    test_item = torch.Tensor(test_item).type(torch.long)

    train_item = train_item + n_user
    test_item = test_item + n_user
    num_nodes = n_user + m_item
    train_adj = SparseTensor(row=train_user, col=train_item, sparse_sizes=(num_nodes, num_nodes))
    test_adj = SparseTensor(row=test_user, col=test_item, sparse_sizes=(num_nodes, num_nodes))

    return n_user, m_item, train_adj, train_unique_users, test_adj, test_unique_users
