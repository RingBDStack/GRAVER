from webbrowser import UnixBrowser
import os
import sys
sys.path.append("/home/shijh25/graver-main/GRAVER")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from models import DGI, GraphCL, Lp,GcnLayers
from layers import GCN, AvgReadout
from layers.disenconv import DisenGCN
import numpy as np
from sklearn.decomposition import PCA
from typing import List
from torch_geometric.datasets import TUDataset,Planetoid,Amazon,Coauthor,Reddit
from torch_geometric.loader import DataLoader
import os
import dgl
from torch_geometric.data import Data
import scipy.sparse as sp
from torch_geometric.utils import dense_to_sparse
import cv2
from utils.data_utils import *

class GraphonGraphGenerator:
    def __init__(self, graphon: np.ndarray, num_nodes: int, token: torch.Tensor):
        self.graphon = graphon
        self.num_nodes = num_nodes
        self.token = token

    def generate_graph(self) -> Data:
        if isinstance(self.graphon, torch.Tensor):
            graphon_np = self.graphon.detach().cpu().numpy()
        elif isinstance(self.graphon, np.ndarray):
            graphon_np = self.graphon
        else:
            raise TypeError(f"Unsupported graphon type: {type(self.graphon)}")
        graphon_resized = cv2.resize(
            graphon_np,
            dsize=(self.num_nodes, self.num_nodes),
            interpolation=cv2.INTER_LINEAR
        )
        sampled_adj = (np.random.rand(self.num_nodes, self.num_nodes) < graphon_resized).astype(np.int32)
        sampled_adj = np.triu(sampled_adj, k=1)
        sampled_adj = sampled_adj + sampled_adj.T
        edge_index = dense_to_sparse(torch.tensor(sampled_adj, dtype=torch.float))[0]
        x = self.token.repeat(self.num_nodes, 1)
        graph_data = Data(x=x, edge_index=edge_index, num_nodes=self.num_nodes)
        return graph_data

class MoE_CoE_Router(nn.Module):
    def __init__(self, num_tokens: int, num_labels_list: List[int], input_dim: int = 64):
        super(MoE_CoE_Router, self).__init__()
        self.num_tokens = num_tokens
        self.num_labels_list = num_labels_list
        self.input_dim = input_dim
        self.moe_weights = nn.Parameter(torch.randn(num_tokens))
        self.coe_weights = nn.ParameterList([
            nn.Parameter(torch.randn(num_labels))
            for num_labels in num_labels_list
        ])

    def forward(self, tokens: torch.Tensor, graphons_list: List[List[np.ndarray]]):
        """
        tokens: Tensor of shape [num_tokens, input_dim]
        graphons_list: List[num_tokens][num_labels_i] -> np.ndarray of shape [5, 5]
        """
        device = tokens.device
        moe_w = F.softmax(self.moe_weights, dim=0)
        final_token = torch.matmul(moe_w, tokens)
        graphons_weighted_per_token = []
        for i, graphons in enumerate(graphons_list):
            coe_w = F.softmax(self.coe_weights[i], dim=0).to(device)
            stacked = torch.stack([
                torch.from_numpy(g).float().to(device)
                for g in graphons
            ], dim=0)
            coe_graphon = torch.einsum('l,lxy->xy', coe_w, stacked)
            graphons_weighted_per_token.append(coe_graphon)
        stacked_graphons = torch.stack(graphons_weighted_per_token, dim=0)
        final_graphon = torch.einsum('t,txy->xy', moe_w.to(device), stacked_graphons)
        return final_token.unsqueeze(0), final_graphon


class downprompt(nn.Module):
    def __init__(self, gen_num_nodes, weights_list, ft_in, nb_classes, type, feature_dim, num_labels_list, num_tokens=4, dropout_rate=0.1):
        super(downprompt, self).__init__()
        self.prefeature = prefeatureprompt(weights_list,dim=feature_dim,type=type, num_labels_list=num_labels_list, num_tokens=num_tokens, dropout_rate=dropout_rate)
        self.nb_classes = nb_classes
        self.leakyrelu = nn.ELU()
        self.one = torch.ones(1,ft_in).cuda()
        self.ave = torch.FloatTensor(nb_classes,ft_in).cuda()
        self.gen_num_nodes = gen_num_nodes

    def forward(self, features, adj, gcn, idx, seq, graphon_list, labels=None, train=0):
        features1, graphon, token = self.prefeature(features, graphon_list)
        graphonGraphGenerator = GraphonGraphGenerator(graphon=graphon, num_nodes=self.gen_num_nodes, token=token)
        num_tar_nodes = len(idx)
        generated_graphs = [graphonGraphGenerator.generate_graph() for _ in range(num_tar_nodes)]
        features1, adj = inject_graphs_return_sparse(generated_graphs=generated_graphs, target_x=features1, target_adj=adj, idx=idx)
        embeds1 = gcn(features1, adj).squeeze()
        pretrain_embs1 = embeds1[idx]      
        rawret = pretrain_embs1
        rawret = rawret.cuda()
        if train == 1:
            self.ave = averageemb(labels=labels, rawret=rawret,nb_class=self.nb_classes)
        ret = torch.FloatTensor(seq.shape[0],self.nb_classes).cuda()
        rawret = torch.cat((rawret,self.ave),dim=0)
        rawret = torch.cosine_similarity(rawret.unsqueeze(1), rawret.unsqueeze(0), dim=-1)
        ret = rawret[:seq.shape[0],seq.shape[0]:]
        ret = F.softmax(ret, dim=1)
        en_result = self.calculate_uncertainty(ret)
        ret = ret.float().cuda()
        en_result = en_result.float().cuda()
        return ret, en_result

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def calculate_uncertainty(self, softmax_output):
        epsilon = 1e-8
        entropy = -torch.sum(softmax_output * torch.log(softmax_output + epsilon), dim=1)
        return entropy

class downstreamprompt(nn.Module):
    def __init__(self,hid_units):
        super(downstreamprompt, self).__init__()
        self.weight= nn.Parameter(torch.FloatTensor(1,hid_units), requires_grad=True)
        self.act = nn.ELU()
        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
    def forward(self, graph_embedding):
        graph_embedding=self.weight * graph_embedding
        return graph_embedding
    
class composedtoken(nn.Module):
    def __init__(self, weights_list, type:str, num_labels_list, num_tokens=4, dim=64):
        super(composedtoken, self).__init__()
        self.texttoken = weights_list
        self.prompt = weighted_prompt(num_tokens, dim, num_labels_list).cuda()
        self.type = type

    def forward(self, seq, graphon_list):
        texttoken, graphon = self.prompt(self.texttoken, graphon_list)
        mix_token = texttoken
        if self.type == 'add':
            texttoken = texttoken.repeat(seq.shape[0],1)
            rets = texttoken + seq
        if self.type == 'mul':
            rets = texttoken * seq
        return rets, graphon, mix_token

class prefeatureprompt(nn.Module):
    def __init__(self, weights_list, dim,type:str, num_labels_list, num_tokens=4, dropout_rate=0.1):
        super(prefeatureprompt, self).__init__()
        self.precomposedfeature = composedtoken(weights_list, type, num_labels_list, num_tokens, dim)
        self.preopenfeature = downstreamprompt(dim)
        self.combineprompt = combineprompt()
    def forward(self, seq, graphon_list):
        seq1, graphon, token = self.precomposedfeature(seq, graphon_list)
        seq2 = self.preopenfeature(seq)
        ret =  self.combineprompt(seq1 ,seq2) 
        return ret, graphon, token

class combineprompt(nn.Module):
    def __init__(self):
        super(combineprompt, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(1, 2), requires_grad=True)
        self.act = nn.ELU()
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
    def forward(self, graph_embedding1, graph_embedding2):
        graph_embedding = self.weight[0][0] * graph_embedding1 + self.weight[0][1] * graph_embedding2
        return self.act(graph_embedding)
    
class weighted_prompt(nn.Module):
    def __init__(self, num_tokens, dim, num_labels_list):
        super(weighted_prompt, self).__init__()
        self.weight= nn.Parameter(torch.FloatTensor(1, num_tokens), requires_grad=True)
        self.act = nn.ELU()
        self.reset_parameters()
        self.routingnet = MoE_CoE_Router(num_tokens=num_tokens, num_labels_list=num_labels_list, input_dim=dim)
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, tokens, graphon_list):
        tokens = self.weight.T * tokens
        token, graphon = self.routingnet(tokens, graphon_list)
        return token, graphon
    
def averageemb(labels,rawret,nb_class):
    retlabel = torch_scatter.scatter(src=rawret,index=labels,dim=0,reduce='mean')
    return retlabel

class PrePrompt(nn.Module):
    def __init__(self, n_in, n_h, init_k, delta_k, routit, tau, dropout, activation, sample, num_layers_num, type, num_tokens=4):
        super(PrePrompt, self).__init__()
        self.lp = Lp(n_in, n_h)
        self.gcn = DisenGCN(n_in, n_h, init_k, delta_k, routit, tau, dropout, num_layers_num)
        self.read = AvgReadout()
        self.prompttype = type
        self.negative_sample = torch.tensor(sample, dtype=int).cuda()
        self.loss = nn.BCEWithLogitsLoss()
        print(num_tokens)
        self.masks_logits = nn.Parameter(torch.randn(num_tokens, n_in))

    def forward(self, seq_list, adj_list):
        mask_prob = torch.sigmoid(self.masks_logits)
        masked_features = [seq * mask_prob[i].unsqueeze(0) for i, seq in enumerate(seq_list)]    
        prelogits = [self.lp(self.gcn, preseq, adj) for preseq, adj in zip(masked_features, adj_list)]
        logits = torch.cat(prelogits, dim=0)
        lploss = compareloss(logits, self.negative_sample, temperature=1)
        total_loss = lploss
        print(f"lploss: {lploss.item()}")
        return total_loss

    def embed(self, seq, adj): 
        h_1 = self.gcn(seq, adj)
        c = self.read(h_1, None)
        return h_1.detach(), c.detach()
    
def mygather(feature, index):
    input_size=index.size(0)
    index = index.flatten()
    index = index.reshape(len(index), 1)
    index = torch.broadcast_to(index, (len(index), feature.size(1)))
    res = torch.gather(feature, dim=0, index=index)
    return res.reshape(input_size,-1,feature.size(1))

def compareloss(feature,tuples,temperature):
    h_tuples=mygather(feature,tuples)
    temp = torch.arange(0, len(tuples))
    temp = temp.reshape(-1, 1)
    temp = torch.broadcast_to(temp, (temp.size(0), tuples.size(1)))
    temp=temp.cuda()
    h_i = mygather(feature, temp)
    sim = F.cosine_similarity(h_i, h_tuples, dim=2)
    exp = torch.exp(sim)
    exp = exp / temperature
    exp = exp.permute(1, 0)
    numerator = exp[0].reshape(-1, 1)
    denominator = exp[1:exp.size(0)]
    denominator = denominator.permute(1, 0)
    denominator = denominator.sum(dim=1, keepdim=True)
    res = -1 * torch.log(numerator / denominator)
    return res.mean()

def pca_compression(seq,k):
    pca = PCA(n_components=k)
    seq = pca.fit_transform(seq) 
    print(pca.explained_variance_ratio_.sum())
    return seq

def prompt_pretrain_sample(adj,n):
    nodenum=adj.shape[0]
    indices=adj.indices
    indptr=adj.indptr
    res=np.zeros((nodenum,1+n))
    whole=np.array(range(nodenum))
    for i in range(nodenum):
        nonzero_index_i_row=indices[indptr[i]:indptr[i+1]]
        zero_index_i_row=np.setdiff1d(whole,nonzero_index_i_row)
        np.random.shuffle(nonzero_index_i_row)
        np.random.shuffle(zero_index_i_row)
        if np.size(nonzero_index_i_row)==0:
            res[i][0] = i
        else:
            res[i][0]=nonzero_index_i_row[0]
        res[i][1:1+n]=zero_index_i_row[0:n]
    return res.astype(int)



