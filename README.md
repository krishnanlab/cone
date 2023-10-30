# Context-specific Network Embedding via Contextualized Graph Attention

## Installation

```bash
conda create -n cone python=3.9 -y && conda activate cone

# Upgrade pip
pip install pip setuptools -U

# Install CUDA enabled packages for CUDA 11.8 (adjust to your system accordingly)
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install torch_geometric==2.4.0 torch_cluster==1.6.3 -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

# Install CONE along with its requirements (in editable mode)
pip install -e .

# Optional steps
pip install -r requirements.txt  # install packages with pinned versioned
conda clean --all -y  # clean up conda environment
```

## Usage notes

### Embedding training

Run CONE embedding training for `PINPPI` network, using GTEx tissue expressed
genes as contexts:

```bash
python main.py network=pinppi context=tissue_gtex_expr
```

After the training is completed, the results can be found under the `dump/`
directory in the run directory. By default, this will be
`outputs/cone-pinppi-tissue_gtex_expr-default/dump`.

### Evaluation

Run the DisGeNET evaluation on the generated embeddings:

```bash
python evaluate_disgenet.py --mode cone --emb_dir outputs/cone-pinppi-tissue_gtex_expr-default/dump/
```

The results will be saved to `results/cone-pinppi-tissue_gtex_expr-default_disgenet.csv`

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
