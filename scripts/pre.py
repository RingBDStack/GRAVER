# main.py (Pretrain)

import os
import sys
import random
import argparse
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset, Planetoid, Amazon, Coauthor, Reddit
from torch_geometric.loader import DataLoader
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import f1_score
from tqdm import tqdm
import wandb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from GRAVER.model import *
from GRAVER.utils import *
from GRAVER.utils.process import *
from GRAVER.utils.data_utils import *
from config import get_args

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def train_model(args):
    datas, num_tokens = get_pretrain_data(args.dataset, args.experiment_type)
    features_list, adj_list = [], []

    for data in datas:
        feature = torch.cat((data.svd_x, data.text_svd_embedding), dim=1)
        features_list.append(feature)
        adj = edge_index_to_sparse_adj(data.edge_index, data.x.shape[0])
        adj_list.append(adj)

    adj_all = combine_dataset(adj_list)
    negetive_sample = prompt_pretrain_sample(adj_all, 50)

    for i in range(len(adj_list)):
        adj_list[i] = sparse_mx_to_torch_sparse_tensor(adj_list[i])

    model = PrePrompt(
        args.disenconv_inp_dim, args.disenconv_hid_dim, args.disenconv_init_k,
        args.disenconv_delta_k, args.disenconv_routit, args.disenconv_tau,
        args.disenconv_dropout, args.nonlinearity, negetive_sample,
        args.disenconv_num_layers, args.combinetype, num_tokens
    )
    model = model.cuda()
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_coef)
    features_list = [feat.cuda() for feat in features_list]
    adj_list = [adj.cuda() for adj in adj_list]
    best = 1e9
    firstbest = 0
    cnt_wait = 0

    for epoch in range(args.nb_epochs):
        set_seed(args.seed)
        model.train()
        optimiser.zero_grad()
        loss = model(features_list, adj_list)
        loss.backward()
        optimiser.step()
        print(f"[Epoch {epoch:03d}] Loss: {loss.item():.4f}")
        wandb.log({"pretrain_loss": loss.item()})
        if loss < best:
            best = loss.item()
            best_t = epoch
            cnt_wait = 0
            firstbest = 1
            torch.save(model.state_dict(), args.save_name)
            print(f"✅ Model saved at epoch {epoch}: {args.save_name}")
        else:
            cnt_wait += 1
        if cnt_wait == args.patience:
            print("⏹️ Early stopping triggered.")
            break
    wandb.log({"final_best_loss": best})
    print(f"🏁 Training finished. Best Loss: {best:.4f} at Epoch {best_t}")

if __name__ == "__main__":
    args = get_args()
    print('-' * 100)
    print(args)
    print('-' * 100)
    set_seed(args.seed)
    device = torch.device("cuda")
    print(f"Using device: {device}")

    wandb.init(
        project="GRAVER-node-finetune",
        config=vars(args),
        settings=wandb.Settings(init_timeout=240)
    )

    train_model(args)
