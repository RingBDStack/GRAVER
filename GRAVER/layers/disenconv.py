import torch
import torch.nn as nn
import torch.nn.functional as F

class DisenGCN(nn.Module):
    def __init__(self, 
                 inp_dim,
                 hid_dim,
                 init_k,
                 delta_k,
                 routit,
                 tau,
                 dropout,
                 num_layers,
                 **kwargs):
        super(DisenGCN, self).__init__()
        self.init_disen = InitDisenLayer(inp_dim, hid_dim, init_k)
        
        self.conv_layers = nn.ModuleList()
        k = init_k
        for l in range(num_layers):
            fac_dim = hid_dim // k
            self.conv_layers.append(RoutingLayer(k, routit, tau))
            inp_dim = fac_dim * k
            k -= delta_k   
        
        self.dropout = dropout
    
    def _dropout(self, X):
        return F.dropout(X, p=self.dropout, training=self.training)
        
    def forward(self, X, edges):
        Z = self.init_disen(X)
        for disen_conv in self.conv_layers:
            Z = disen_conv(Z, edges)
            Z = self._dropout(torch.relu(Z))
        return Z.reshape(len(Z), -1)

class InitDisenLayer(nn.Module):
    def __init__(self, inp_dim, hid_dim, num_factors):
        super(InitDisenLayer, self).__init__()

        self.inp_dim = inp_dim
        self.hid_dim = (hid_dim // num_factors) * num_factors
        self.num_factors = num_factors
        
        self.factor_lins = nn.Linear(self.inp_dim, self.hid_dim)
        
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, X):
        Z = self.factor_lins(X).view(-1, self.num_factors, self.hid_dim // self.num_factors)
        Z = F.normalize(torch.relu(Z), dim=2)
        return Z

class RoutingLayer(nn.Module):
    def __init__(self, num_factors, routit, tau):
        super(RoutingLayer, self).__init__()
        self.num_factors = num_factors
        self.routit = routit
        self.tau = tau

    def forward(self, x, edges):
        edges = edges.coalesce()
        src = edges.indices()[0]
        trg = edges.indices()[1]
        n, k, delta_d = x.shape

        z = x
        c = x

        for t in range(self.routit):
            p = (z[trg] * c[src]).sum(dim=2, keepdim=True)
            p = F.softmax(p / self.tau, dim=1)
            weight_sum = (p * z[trg])
            c = z + torch.zeros_like(z).index_add_(0, src, weight_sum)
            c = F.normalize(c, dim=2)
        return c
