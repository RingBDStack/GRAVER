import argparse
from datetime import datetime
import os

def get_args():
    parser = argparse.ArgumentParser("GRAVER")
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    parser.add_argument("--dataset", type=str, default="cora", help="data")
    parser.add_argument("--seed", type=int, default=39, help="seed")
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument("--experiment_type", type=str, default="cross-dataset",help="experiment_type")
    # train_params
    parser.add_argument("--lr", type=float, default=0.00001, help="Learning rate")  #
    parser.add_argument("--l2_coef", type=float, default=0.0, help="pre_weight_decay")
    parser.add_argument(
        "--hid_units", type=int, default=256, help="GCN output dimension"
    )
    parser.add_argument(
        "--lambda_entropy",
        type=float,
        default=0.20401015296835048,
        help="entorpy_loss_weight",
    )
    parser.add_argument("--downstreamlr", type=float, default=1e-3, help="downstream learning rate")
    parser.add_argument(
        "--combinetype", type=str, default="mul", help="the type of text combining"
    )
    parser.add_argument(
        "--model_path", type=str, default="unwork", help="be helpful in down only"
    )
    parser.add_argument(
        "--nb_epochs", type=int, default="10000", help="pretrain_num_epochs"
    )
    parser.add_argument("--shot_num", type=int, default="1", help="few_shot_num")
    parser.add_argument(
        "--fw_epochs", type=int, default="1", help="fewshot_num_epochs" 
    )
    parser.add_argument("--prompt_times", type=int, default="20", help="total_avg") #50
    # disen_params
    parser.add_argument("--disenconv_init_k", type=int, default=2, help="Initial number of factors")
    parser.add_argument("--disenconv_delta_k", type=int, default=0, help="Number of factors reduced per layer")
    parser.add_argument("--disenconv_routit", type=int, default=1, help="Number of routing iterations")
    parser.add_argument("--disenconv_tau", type=float, default=1.0, help="Softmax temperature parameter")
    parser.add_argument("--disenconv_dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--disenconv_num_layers", type=int, default=1, help="Number of DisenGCN layers")
    parser.add_argument("--groups", type=int, default=20, help="Temporary parameter: although 200 groups are prepared, only 100 are selected")
    parser.add_argument("--test_num_nodes", type=int, default=300, help="Number of randomly selected test nodes")
    args = parser.parse_args()
    args.unify_dim = 64
    args.disenconv_inp_dim = args.unify_dim
    args.disenconv_hid_dim = args.hid_units
    args.patience = 50
    args.LP = False
    args.nonlinearity = "prelu"
    save_dir = "../graver-main/data/save_model"
    os.makedirs(save_dir, exist_ok=True)
    save_name = f"{args.dataset}_{current_time}.pkl"
    save_path = os.path.join(save_dir, save_name)
    args.save_name = save_path
    args.data_path = "../graver-main/data/ori_data"
    args.few_shot_data_dir = f"../graver-main/data/few_data_graph/{args.dataset}/{args.shot_num}-shot"
    args.save_dir = save_dir
    args.text_feature_dim = 768
    args.text_feature_out_dim = 32
    args.gen_num_nodes = 10
    args.test_dir = f'../graver-main/data/test_split/graph'
    return args