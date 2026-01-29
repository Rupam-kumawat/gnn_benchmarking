



# gprgnn_models.py

import torch
import torch.nn.functional as F
import numpy as np

from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch.nn import Parameter
from torch.nn import Linear
from torch_geometric.nn import MessagePassing, APPNP

class GPR_prop(MessagePassing):
    '''
    Propagation class for GPRGNN
    '''
    def __init__(self, K, alpha, Init, Gamma=None, bias=True, **kwargs):
        super(GPR_prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.Init = Init
        self.alpha = alpha

        assert Init in ['SGC', 'PPR', 'NPPR', 'Random', 'WS']
        if Init == 'SGC':
            TEMP = 0.0 * np.ones(K + 1)
            TEMP[alpha] = 1.0
        elif Init == 'PPR':
            TEMP = alpha * (1 - alpha)**np.arange(K + 1)
            TEMP[-1] = (1 - alpha)**K
        elif Init == 'NPPR':
            TEMP = (alpha)**np.arange(K + 1)
            TEMP = TEMP / np.sum(np.abs(TEMP))
        elif Init == 'Random':
            bound = np.sqrt(3 / (K + 1))
            TEMP = np.random.uniform(-bound, bound, K + 1)
            TEMP = TEMP / np.sum(np.abs(TEMP))
        elif Init == 'WS':
            TEMP = Gamma

        self.temp = Parameter(torch.tensor(TEMP))

    def reset_parameters(self):
        torch.nn.init.zeros_(self.temp)
        if self.Init == 'PPR':
            for k in range(self.K + 1):
                self.temp.data[k] = self.alpha * (1 - self.alpha)**k
            self.temp.data[-1] = (1 - self.alpha)**self.K
        elif self.Init == 'SGC':
            self.temp.data[self.alpha] = 1.0


    def forward(self, x, edge_index, edge_weight=None):
        edge_index, norm = gcn_norm(
            edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)

        hidden = x * (self.temp[0])
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            gamma = self.temp[k + 1]
            hidden = hidden + gamma * x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

class GPRGNN(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, dprate, K, alpha, Init, ppnp):
        super(GPRGNN, self).__init__()
        self.lin1 = Linear(nfeat, nhid)
        self.lin2 = Linear(nhid, nclass)

        if ppnp == 'PPNP':
            self.prop1 = APPNP(K, alpha)
        elif ppnp == 'GPR_prop':
            self.prop1 = GPR_prop(K, alpha, Init)

        self.dprate = dprate
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.prop1.reset_parameters()

    def forward(self, x, edge_index):
        x = self.get_embeddings(x) # Get MLP output
        
        if self.dprate > 0.0:
            x = F.dropout(x, p=self.dprate, training=self.training)
            
        x = self.prop1(x, edge_index)
        return x

    def get_embeddings(self, x):
        """Returns the output of the MLP transformation part."""
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return x
