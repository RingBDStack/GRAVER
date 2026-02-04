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
from GRAVER.utils import process
from GRAVER.utils.data_utils import *
from GRAVER.model import PrePrompt, downprompt

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def train_model(args):
    num_tokens = get_num_tokens(args.dataset, args.experiment_type)
    model = PrePrompt(
        args.disenconv_inp_dim, args.disenconv_hid_dim, args.disenconv_init_k,
        args.disenconv_delta_k, args.disenconv_routit, args.disenconv_tau,
        args.disenconv_dropout, args.nonlinearity, 1, args.disenconv_num_layers,
        args.combinetype, num_tokens
    ).cuda()

    print('#' * 50)
    print('Downstream dataset:', args.dataset)

    num_labels_list = [get_num_lables(pre_data_name) for pre_data_name in get_pre_datas(args.dataset, args.experiment_type)]
    graphon_list = [
        [torch.load(path) for path in get_save_graphon_path(args.dataset, args.experiment_type, pre_data_name)]
        for pre_data_name in get_pre_datas(args.dataset, args.experiment_type)
    ]

    data = get_data(args.dataset)
    features = torch.cat((data.svd_x, data.text_svd_embedding), dim=1).cuda()
    adj = edge_index_to_sparse_adj(data.edge_index, data.x.shape[0])
    sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj).cuda()
    labels = data.y.squeeze()
    nb_classes = len(torch.unique(labels))
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found at: {args.model_path}")
    model.load_state_dict(torch.load(args.model_path))
    with torch.no_grad():
        embeds, _ = model.embed(features, sp_adj)
        embeds = embeds.detach()
    xent = nn.CrossEntropyLoss()
    downstreamlr = args.downstreamlr
    print("-" * 100)
    print("Shot number:", args.shot_num)

    if hasattr(model, "masks_logits"):
        soft_masks = torch.sigmoid(model.masks_logits).detach()
    else:
        raise ValueError("Model has no attribute `masks_logits`")

    group_results = []

    for i in tqdm(range(args.groups)):
        idx_test = torch.load(f'{args.test_dir}/eval_{args.dataset}_{args.experiment_type}_{args.shot_num}shot_group_{i}.pt')
        test_lbls = labels[idx_test].cuda()
        test_embs = embeds[idx_test]

        log = downprompt(
            args.gen_num_nodes, soft_masks, args.hid_units, nb_classes,
            args.combinetype, args.unify_dim, num_labels_list, num_tokens
        ).cuda()

        idx_train = torch.load(f"{args.few_shot_data_dir}/idx/{i}_idx.pt").long().cuda()
        train_lbls = torch.load(f"{args.few_shot_data_dir}/label/{i}_labels.pt").long().squeeze().cuda()
        pretrain_embs = embeds[idx_train]

        opt = torch.optim.Adam(log.parameters(), lr=downstreamlr)

        for epoch in range(args.fw_epochs):
            log.train()
            opt.zero_grad()
            logits, entropy_logits = log(features, sp_adj, model.gcn, idx_train, pretrain_embs, graphon_list, train_lbls, 1)
            loss = xent(logits, train_lbls) + args.lambda_entropy * torch.mean(entropy_logits)
            loss.backward()
            opt.step()
            print(f"[Group {i}] Epoch {epoch+1}: Loss = {loss.item():.4f}")

        log.eval()
        with torch.no_grad():
            logits_test, _ = log(features, sp_adj, model.gcn, idx_test, test_embs, graphon_list)
            acc = (torch.argmax(logits_test, dim=1) == test_lbls).float().mean().item()
        group_results.append(acc)
        print(f"[Group {i}] Final Accuracy: {acc:.4f}")

    mean_acc = np.mean(group_results)
    std_acc = np.std(group_results)
    print(f"\nAverage Accuracy over {args.groups} groups: {mean_acc:.4f}")
    print(f"Standard Deviation: {std_acc:.4f}")

def main():
    args = get_args()
    set_seed(args.seed)
    model_paths = {
        ("cross-dataset", "cora"): "cora.pkl",
        ("cross-dataset", "citeseer"): "citeseer.pkl",
        ("cross-dataset", "pubmed"): "pubmed.pkl",
        ("cross-dataset", "P-tech"): "P-tech.pkl",
        ("cross-domain", "arxiv"): "arxiv.pkl",
        ("cross-domain", "P-home"): "P-home.pkl",
        ("cross-domain", "wikics"): "wikics.pkl",
        ("cross-domain", "P-tech"): "P-tech.pkl",
    }
    key = (args.experiment_type, args.dataset)
    if key not in model_paths:
        raise ValueError("Unsupported experiment_type or dataset")
    args.model_path = os.path.join("../graver-main/data/save_model", model_paths[key])

    wandb.init(
        settings=wandb.Settings(init_timeout=240),
        project="GRAVER-node-finetune",
        config=vars(args)
    )

    print('-' * 100)
    print(args)
    print('-' * 100)
    train_model(args)

if __name__ == "__main__":
    main()
