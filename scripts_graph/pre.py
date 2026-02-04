import os
import sys
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import wandb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import get_args
from GRAVER_graph.model import *
from GRAVER_graph.utils.process import *
from GRAVER_graph.utils.data_utils import *

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
    negative_sample = prompt_pretrain_sample(adj_all, 50)
    adj_list = [sparse_mx_to_torch_sparse_tensor(adj) for adj in adj_list]

    model = PrePrompt(
        args.disenconv_inp_dim, args.disenconv_hid_dim, args.disenconv_init_k,
        args.disenconv_delta_k, args.disenconv_routit, args.disenconv_tau,
        args.disenconv_dropout, args.nonlinearity, negative_sample,
        args.disenconv_num_layers, args.combinetype, num_tokens
    ).cuda()

    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_coef)
    features_list = [feat.cuda() for feat in features_list]
    adj_list = [adj.cuda() for adj in adj_list]

    best_loss = float('inf')
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

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_epoch = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), args.save_name)
            print(f"✅ Model saved at epoch {epoch}: {args.save_name}")
        else:
            cnt_wait += 1

        if cnt_wait == args.patience:
            print("⏹️ Early stopping triggered.")
            break

    wandb.log({"final_best_loss": best_loss})
    print(f"🏁 Training finished. Best Loss: {best_loss:.4f} at Epoch {best_epoch}")

if __name__ == "__main__":
    args = get_args()
    print('-' * 100)
    print(args)
    print('-' * 100)
    set_seed(args.seed)

    wandb.init(
        project="GRAVER-graph-pretrain",
        config=vars(args),
        settings=wandb.Settings(init_timeout=240)
    )

    train_model(args)
