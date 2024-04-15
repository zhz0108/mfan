# MFAN
Source Code Implementation of MFAN (Multi-Faceted Anchor Nodes Attack)

## Python Environment Requirement
The required conda environment is [here](environment.yml)

## Usage

To train and test for MFAN, run
```
python run.py --dataset [dataset] --K [K] --xi [xi]
```

To evaluate MFAN on transfer models, run
```
python attack_gat.py --dataset [dataset] --K [K] --xi [xi]
```
and
```
python attack_node2vec.py --dataset [dataset] --K [K] --xi [xi]
```
