from model.GONN import GONN
import torch
import torch.nn as nn

class OrdRec(nn.Module):
    def __init__(self, params, num_users, num_items):
        super(OrdRec, self).__init__()

        self.params = params
        self.num_users = num_users
        self.num_items = num_items
        self.num_nodes = self.num_users + self.num_items

        self.GONN = GONN(self.params)
        self.x = nn.Parameter(torch.empty(self.num_nodes, self.params['in_channel'], dtype=torch.float))
        #self.embedding = nn.Embedding(self.num_nodes, self.params['in_channel'])

        self.init_embedding()

    def init_embedding(self):
        nn.init.xavier_uniform_(self.x)

    def forward(self, user_idx, edge_index):
        if self.train():
            assert torch.all(user_idx < self.num_users)
        all_embedding = self.GONN(self.x, edge_index)

        #user_embedding = embedding[:self.num_users]
        user_embedding = all_embedding[user_idx]
        item_embedding = all_embedding[self.num_users:]

        return user_embedding, item_embedding