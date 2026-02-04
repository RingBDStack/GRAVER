from torch_geometric.datasets import TUDataset, Planetoid, Amazon, Coauthor, Reddit
from torch_geometric.loader import DataLoader
import os
import wandb
import dgl
from torch_geometric.data import Data
import torch
import scipy.sparse as sp
import numpy as np
from torch_geometric.utils import dense_to_sparse, to_undirected, to_dense_adj
from typing import List, Tuple

def inject_graphs_return_sparse(
    generated_graphs: List[Data],
    target_x: torch.Tensor,
    target_adj: torch.Tensor,
    idx: List[int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert len(generated_graphs) == len(idx)
    device = target_x.device

    new_x = target_x.clone()
    N = target_x.size(0)

    row, col = target_adj.coalesce().indices() if target_adj.is_sparse else torch.nonzero(target_adj, as_tuple=True)
    row = row.tolist()
    col = col.tolist()
    
    for graph, tgt_node in zip(generated_graphs, idx):
        if graph.edge_index.numel() == 0:
            continue

        graph = graph.to(device)
        num_nodes = graph.num_nodes
        deg = torch.bincount(graph.edge_index[0], minlength=num_nodes)
        max_deg_node = torch.argmax(deg).item()

        new_node_indices = [j for j in range(num_nodes) if j != max_deg_node]
        n_add = len(new_node_indices)
        if n_add == 0:
            continue

        old2new = {old: N + i for i, old in enumerate(new_node_indices)}
        new_x = torch.cat([new_x, graph.x[new_node_indices]], dim=0)

        for s, t in graph.edge_index.t().tolist():
            s_new = tgt_node if s == max_deg_node else old2new.get(s)
            t_new = tgt_node if t == max_deg_node else old2new.get(t)
            if s_new is not None and t_new is not None:
                row.extend([s_new, t_new])
                col.extend([t_new, s_new])

        N += n_add

    edge_index = torch.tensor([row, col], dtype=torch.long, device=device)
    edge_values = torch.ones(edge_index.size(1), dtype=torch.float32, device=device)
    new_adj = torch.sparse_coo_tensor(edge_index, edge_values, size=(new_x.size(0), new_x.size(0)))

    return new_x, new_adj

def edge_index_to_sparse_adj(edge_index, num_nodes):
    row = edge_index[0].numpy()
    col = edge_index[1].numpy()
    data = np.ones(len(row))
    adj_matrix = sp.csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    return adj_matrix

data_names = ['cora', 'citeseer', 'pubmed', 'P-tech', 'P-home', 'wikics', 'arxiv']

def get_data(data_name):
    config = wandb.config
    data_path = config.data_path
    data = torch.load(f'{data_path}/{data_name}.pt')
    return data

def get_num_tokens(data_name, experiment_type='cross-dataset'):
    if experiment_type == 'cross-dataset':
        if data_name in ['cora', 'citeseer', 'pubmed']:
            return 4
        elif data_name == 'P-tech':
            return 5
        else:
            raise ValueError("Invalid dataset.")
    elif experiment_type == 'cross-domain':
        if data_name in ['P-home', 'P-tech', 'wikics']:
            return 4
        elif data_name == 'arxiv':
            return 3
        else:
            raise ValueError("Invalid dataset.")
    else:
        raise ValueError("Invalid experiment_type.")

def get_save_graphon_path(dataset: str, experiment_type: str, pre_data_name: str):
    save_dir = f'/home/shijh25/disen-v2/data/graphon/{experiment_type}/{dataset}/{pre_data_name}'
    os.makedirs(save_dir, exist_ok=True)
    num_labels = get_num_lables(pre_data_name)
    return [f'{save_dir}/label_{i}_graphon.pt' for i in range(num_labels)]

def get_pretrain_data(data_name, experiment_type='cross-dataset'):
    datas = []
    num_tokens = 0
    if experiment_type == 'cross-dataset':
        if data_name == 'cora':
            datas = [get_data(n) for n in ['citeseer', 'pubmed', 'P-home', 'wikics']]
            num_tokens = 4
        elif data_name == 'citeseer':
            datas = [get_data(n) for n in ['cora', 'pubmed', 'P-home', 'wikics']]
            num_tokens = 4
        elif data_name == 'pubmed':
            datas = [get_data(n) for n in ['cora', 'citeseer', 'P-home', 'wikics']]
            num_tokens = 4
        elif data_name == 'P-tech':
            datas = [get_data(n) for n in ['cora', 'citeseer', 'pubmed', 'P-home', 'wikics']]
            num_tokens = 5
    elif experiment_type == 'cross-domain':
        if data_name in ['P-home', 'P-tech']:
            datas = [get_data(n) for n in ['cora', 'citeseer', 'pubmed', 'wikics']]
            num_tokens = 4
        elif data_name == 'wikics':
            datas = [get_data(n) for n in ['cora', 'citeseer', 'pubmed', 'P-home']]
            num_tokens = 4
        elif data_name == 'arxiv':
            datas = [get_data(n) for n in ['P-home', 'P-tech', 'wikics']]
            num_tokens = 3
    return datas, num_tokens

def sparse_to_edges(sparse_matrix):
    coo_matrix = sparse_matrix.tocoo()
    src = coo_matrix.row
    trg = coo_matrix.col
    return (src, trg)

def get_num_lables(dataset: str):
    label_dict = {
        "cora": 7,
        "citeseer": 6,
        "pubmed": 3,
        "P-home": 5,
        "P-tech": 3,
        "wikics": 10,
        "arxiv": 40
    }
    if dataset not in label_dict:
        raise ValueError(f"Invalid dataset: {dataset}")
    return label_dict[dataset]

def get_pre_datas(dataset: str, experiment_type: str):
    if experiment_type == 'cross-domain':
        if dataset == 'arxiv':
            return ['P-home', 'P-tech', 'wikics']
        elif dataset in ['P-home', 'P-tech']:
            return ['citeseer', 'cora', 'pubmed', 'wikics']
        elif dataset == 'wikics':
            return ['citeseer', 'cora', 'pubmed', 'P-home']
        else:
            raise ValueError(f"Invalid dataset: {dataset}")
    elif experiment_type == 'cross-dataset':
        if dataset == 'cora':
            return ['citeseer', 'P-home', 'wikics', 'pubmed']
        elif dataset == 'citeseer':
            return ['cora', 'P-home', 'wikics', 'pubmed']
        elif dataset == 'pubmed':
            return ['cora', 'citeseer', 'P-home', 'wikics']
        elif dataset == 'P-tech':
            return ['cora', 'citeseer', 'P-home', 'wikics', 'pubmed']
        else:
            raise ValueError(f"Invalid dataset: {dataset}")
    else:
        raise ValueError(f"Invalid experiment_type: {experiment_type}")
