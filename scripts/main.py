# GRAVER: Full Training Pipeline (Pretrain + Finetune)

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
from GRAVER.model import *
from GRAVER.utils import *
from GRAVER.utils.process import *
from GRAVER.utils.data_utils import *

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def pretrain(args):
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
    ).cuda()

    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_coef)
    features_list = [feat.cuda() for feat in features_list]
    adj_list = [adj.cuda() for adj in adj_list]

    best = 1e9
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
            torch.save(model.state_dict(), args.save_name)
            print(f"✅ Model saved at epoch {epoch}: {args.save_name}")
        else:
            cnt_wait += 1

        if cnt_wait == args.patience:
            print("⏹️ Early stopping triggered.")
            break

    wandb.log({"final_best_loss": best})
    print(f"🏁 Pretraining finished. Best Loss: {best:.4f} at Epoch {best_t}")

def finetune(args):
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
    sp_adj = sparse_mx_to_torch_sparse_tensor(adj).cuda()
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

    wandb.init(
        project="GRAVER-full-pipeline",
        config=vars(args),
        settings=wandb.Settings(init_timeout=240)
    )

    print('-' * 100)
    print(args)
    print('-' * 100)

    # Step 1: Run pretraining and save model
    pretrain(args)

    # Step 2: Load the saved model directly into finetuning
    args.model_path = args.save_name

    # Step 3: Run downstream finetuning
    finetune(args)


if __name__ == "__main__":
    main()
