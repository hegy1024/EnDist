# EnDist

Code for the paper "Enhancing Graph Neural Network Explainers Using a Distribution Shift Consistency-Guided Graph Generator"

## Installation
Requirments:
- python version 3.9.0 (**note**: the version of python should remain $\ge$ 3.8.0)
- CUDA version 11.7
- torch version 1.13.0
- torch_geometric (pyg) version 2.3.0

To install other packages, please use: 
```sh
pip install -r requirements.txt
```

## Availability

### Datasets

Four real-world graph classification datasets are employed to evaluate EnDist framework: 

- `MUTAG`
- `Benzene`
- `Alkane-Carbonyl`
- `Fluoride-Carbonyl`

All datasets are given in the `data.rar` file.

### Methods

- Baselines

  We compare EnDist with three distribution shift adjusted methods: `MixUpExplainer`, `ProxyExplainer`, `CGExplainer`

- Backbones

  Three post-hoc explanation methods are used as backbones: `GNNExplainer`, `PGExplainer`, `KFactExplainer`

## Usage

To reproduce the results on **one** dataset (data_name: `mutag`, `benz`, `car1`, `car2`) and **one** backbone (backbone_name: `gnne`, `pge`, `kfact`), please run:

```sh
python main.py --mode "ed" --data [data_name] --backbone [backbone_name] --device [device_id] --read_configs --save_params
```

To reproduce the results on **all** datasets and **all** backbones, please run: 

```sh 
bash run.sh [device_id]
```
