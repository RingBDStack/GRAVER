import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from functools import partial
import numpy as np
import tqdm

class Lp(nn.Module):
    def __init__(self, n_in, n_h):
        super(Lp, self).__init__()
        self.sigm = nn.ELU()
        self.act=torch.nn.LeakyReLU()
        self.prompt = nn.Parameter(torch.FloatTensor(1, n_h), requires_grad=True)
        self.reset_parameters()



    def forward(self,gcn,seq,adj):
        ret = gcn(seq, adj)
        ret = self.sigm(ret.squeeze(dim=0))
        return ret

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.prompt)
