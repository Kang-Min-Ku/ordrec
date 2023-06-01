from model.layer import ONGNNConv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Linear, LayerNorm

class GONN(Module):
    def __init__(self, params, num_users, num_items):
        super().__init__()
        self.params = params
        self.num_users = num_users
        self.num_items = num_items
        self.num_nodes = self.num_users+self.num_items
         
        self.x = nn.Embedding(self.num_nodes, self.params['in_channel'])
        
        self.linear_trans_in = ModuleList()
        self.linear_trans_out = Linear(params['hidden_channel'], params['out_channel'])
        self.norm_input = ModuleList()
        self.convs = ModuleList()

        self.tm_norm = ModuleList()
        self.tm_net = ModuleList()

        self.linear_trans_in.append(Linear(params['in_channel'], params['hidden_channel']))

        self.norm_input.append(LayerNorm(params['hidden_channel']))

        for i in range(params['num_layers_input']-1):
            self.linear_trans_in.append(Linear(params['hidden_channel'], params['hidden_channel']))
            self.norm_input.append(LayerNorm(params['hidden_channel']))

        if params['global_gating']==True:
            tm_net = Linear(2*params['hidden_channel'], params['chunk_size'])

        for i in range(params['num_layers']):
            self.tm_norm.append(LayerNorm(params['hidden_channel']))
            
            if params['global_gating']==False:
                self.tm_net.append(Linear(2*params['hidden_channel'], params['chunk_size']))
            else:
                self.tm_net.append(tm_net)
            
            if params['model']=="ONGNN":
                self.convs.append(ONGNNConv(tm_net=self.tm_net[i], tm_norm=self.tm_norm[i], params=params))

        self.params_conv = list(set(list(self.convs.parameters())+list(self.tm_net.parameters())))
        self.params_others = list(self.linear_trans_in.parameters())+list(self.linear_trans_out.parameters())

    def forward(self, user_idx, edge_index):
        # print(self.x[user_idx[:5]])
        print(self.x(user_idx[:5]))
        check_signal = []
        
        if self.train():
            assert torch.all(user_idx < self.num_users)

        x = self.x(torch.arange(self.num_nodes))

        for i in range(len(self.linear_trans_in)):
            x = F.dropout(x, p=self.params['dropout_rate'], training=self.training)
            x = F.relu(self.linear_trans_in[i](x))
            x = self.norm_input[i](x)

        tm_signal = x.new_zeros(self.params['chunk_size'])

        for j in range(len(self.convs)):
            if self.params['dropout_rate2']!='None':
                x = F.dropout(x, p=self.params['dropout_rate2'], training=self.training)
            else:
                x = F.dropout(x, p=self.params['dropout_rate'], training=self.training)
            x, tm_signal = self.convs[j](x, edge_index, last_tm_signal=tm_signal)
            check_signal.append(dict(zip(['tm_signal'], [tm_signal])))

        x = F.dropout(x, p=self.params['dropout_rate'], training=self.training)
        x = self.linear_trans_out(x)
        
        user_embedding = x[user_idx]
        item_embedding = x[self.num_users:]

        #encode_values = dict(zip(['x', 'check_signal'], [x, check_signal]))
        
        return user_embedding, item_embedding
    
    def train_batch(self, user_idx, edge_index):
        user_embedding, item_embedding = self.forward(user_idx, edge_index)
        return torch.matmul(user_embedding, item_embedding.t())

    def rating(self):
        user_embedding = self.x[:self.num_users]
        item_embedding = self.x[self.num_users:]
        return torch.matmul(user_embedding, item_embedding.t())
