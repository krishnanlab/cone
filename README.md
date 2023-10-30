# Context-specific Network Embedding via Contextualized Graph Attention

## Installation

```bash
conda create -n cone python=3.9 -y && conda activate cone

# Upgrade pip
pip install pip setuptools -U

# Install CUDA enabled packages for CUDA 11.8 (adjust to your system accordingly)
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install torch_geometric==2.4.0 torch_cluster==1.6.3 -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

# Install rest of the dependencies
pip install -r requirements.txt

# Clean up (optional)
conda clean --all -y
```

## Usage notes

Run CONE embedding training for `PINPPI` network, using GTEx tissue expressed
genes as contexts.

```bash
main.py network=pinppi context=tissue_gtex_expr
```

## Cite our work

```bibtex
@article{liu2023cone,
  title={CONE: COntext-specific Network Embedding via Contextualized Graph Attention},
  author={Liu, Renming and Yuan, Hao and Johnson, Kayla A and Krishnan, Arjun},
  journal={bioRxiv},
  pages={2023--10},
  year={2023},
  publisher={Cold Spring Harbor Laboratory}
}
```
