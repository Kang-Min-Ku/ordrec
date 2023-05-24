from model.GONN import GONN
import torch.nn as nn

class OrdRec():
    def __init__(self, num_users, num_items, params):
        self.params = params
        self.num_users = num_users
        self.num_items = num_items
        self.num_nodes = self.num_users + self.num_items

        self.GONN = GONN(self.params)
        self.embedding = nn.Embedding(self.num_nodes, self.params['in_channel']) #need initialze

    def init_embedding(self):
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, node_idx, edge_index):
        x = self.embedding(node_idx)
        encode_values = self.GONN(x, edge_index)
        return encode_values