# GRAVER: Generative Graph Vocabularies for Robust Graph Foundation Models Fine-tuning

This repository contains the official implementation of **GRAVER**, proposed in the paper *GRAVER: Generative Graph Vocabularies for Robust Graph Foundation Models Fine-tuning* (NeurIPS 2025), which is a robust and efficient fine-tuning framework for **Graph Foundation Models (GFMs)** under **few-shot cross-domain settings**.

## 1. Environment Requirements

### Core Dependency Versions

The following packages are required to run the project:

* `torch == 1.10.1+cu113`
* `torch-geometric == 2.5.3`
* `torch-cluster == 1.5.9`
* `torch-scatter == 2.0.9`
* `torch-sparse == 0.6.12`
* `torch-spline-conv == 1.2.1`
* `scikit-learn == 1.3.2`
* `numpy == 1.24.4`
* `pandas == 2.0.3`
* `matplotlib == 3.7.5`
* `scipy == 1.10.1`
* `huggingface-hub == 0.29.3`
* `wandb == 0.19.8`
* `tqdm == 4.67.1`
* `protobuf == 5.29.3`
* `requests == 2.32.3`
* `typing_extensions == 4.12.2`
* `attrs == 25.2.0`
* `dgl == 0.5.0`

To install all dependencies:

```bash
pip install -r requirements.txt
```

---

## 2. Dataset Download & Organization

You can download the full dataset from HuggingFace:

📦 [Click to Download GRAVER Dataset](https://huggingface.co/datasets/aboutime233/GRAVER-NIPS25-datasets/resolve/main/data.zip?download=true)

Then unzip and organize it under the `data/` directory as follows:

```
data/
├── ori_data/                 # Original graph data
├── save_model/               # Pretrained models
├── test_split/graph/         # Test indices for graph classification
├── few_data/                 # Few-shot data for node classification
└── few_data_graph/           # Few-shot data for graph classification
```

---

## 3. Quick Start

### Node Classification

Run a 1-shot cross-dataset node classification task:

```bash
cd scripts
python main.py --dataset cora --shot_num 1 --experiment_type cross-dataset
```

### Graph Classification

Run a 1-shot cross-domain graph classification task:

```bash
cd scripts_graph
python main.py --dataset wikics --shot_num 1 --experiment_type cross-domain
```

---

## 4. Full Pipeline: Pretraining + Finetuning

To run the full pretrain-then-finetune pipeline:

```bash
python main.py --dataset cora --shot_num 1 --experiment_type cross-dataset
```

---

## 5. Recommended Settings to Reproduce Results

### Node Classification (e.g., Cora)

```bash
python main.py --dataset cora --shot_num 1 --experiment_type cross-dataset
```

**Recommended parameters:**

```bash
--lr 0.00001 
--hid_units 256 
--lambda_entropy 0.204 
--downstreamlr 0.001
```

### Graph Classification (e.g., Wiki-CS)

```bash
python main.py --dataset wikics --shot_num 1 --experiment_type cross-domain
```

**Recommended parameters:**

```bash
--lr 0.00001 
--hid_units 256 
--lambda_entropy 0.078 
--downstreamlr 0.001
```

---

## 6. Project Structure

```
GRAVER/
├── scripts/                   # Node classification scripts
├── scripts_graph/             # Graph classification scripts
├── GRAVER/                    # Node classification model code
├── GRAVER_graph/              # Graph classification model code
├── config.py                  # Central configuration
└── requirements.txt           # Dependency list
```

---

## 7. Using wandb (Optional)

To enable experiment tracking with wandb:

```bash
wandb login [YOUR_API_KEY]
```

Metrics tracked:

* Training loss & accuracy
* Top-20 group accuracy rankings
* Path of best saved pretrained model

---

## 8. Full Argument List

Here are the supported command-line arguments:

| Argument                 | Description                                       | Default           |
| ------------------------ |---------------------------------------------------| ----------------- |
| `--dataset`              | Target dataset name                               | `"cora"`          |
| `--seed`                 | Random seed                                       | `39`              |
| `--gpu`                  | GPU index                                         | `0`               |
| `--experiment_type`      | `cross-dataset` or `cross-domain`                 | `"cross-dataset"` |
| `--lr`                   | Learning rate (pre-train)                         | `0.00001`         |
| `--l2_coef`              | L2 regularization                                 | `0.0`             |
| `--hid_units`            | Hidden dimension                                  | `256`             |
| `--lambda_entropy`       | Entropy loss weight                               | `0.204`           |
| `--downstreamlr`         | Learning rate (fine-tune)                         | `0.001`           |
| `--combinetype`          | Feature combination strategy (`add`, `mul`, etc.) | `"mul"`           |
| `--model_path`           | Path to load pre-trained model                    | `"unwork"`        |
| `--nb_epochs`            | Pre-training epochs                               | `10000`           |
| `--shot_num`             | Few-shot samples per class                        | `1`               |
| `--fw_epochs`            | Fine-tuning epochs                                | `1`               |
| `--prompt_times`         | Number of prompt averaging times                  | `20`              |
| `--disenconv_init_k`     | Initial factor count                              | `2`               |
| `--disenconv_delta_k`    | Factors reduced per layer                         | `0`               |
| `--disenconv_routit`     | Routing iterations                                | `1`               |
| `--disenconv_tau`        | Temperature for softmax                           | `1.0`             |
| `--disenconv_dropout`    | Dropout rate                                      | `0.2`             |
| `--disenconv_num_layers` | Number of DisenGCN layers                         | `1`               |
| `--groups`               | Number of candidate groups                        | `20`              |
| `--test_num_nodes`       | Number of test nodes                              | `300`             |

For more examples and help, please refer to the `scripts/` and `scripts_graph/` folders.

## 9. Citation

If you find this work useful, please cite:

```
@inproceedings{yuan2025graver,
  author    = {Yuan, Haonan and Sun, Qingyun and Shi, Junhua and Fu, Xingcheng and Hooi, Bryan and Li, Jianxin and Yu, Philip S},
  title     = {{GRAVER}: Generative Graph Vocabularies for Robust Graph Foundation Models Fine-tuning},
  booktitle = {The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year      = {2025}
}
```

---

## 📬 Contact

For questions or discussions, please contact **[Haonan Yuan](mailto:yuanhn@buaa.edu.cn)** or open an issue in this repository.
