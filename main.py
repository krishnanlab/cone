from contextlib import contextmanager

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch_geometric.data import Data

from cone.io import dump_ctxt_nets
from cone.loaders import load_data
from cone.model import CONE
from cone.trainer import Trainer
from cone.utils import count_params, get_device


def setup_model(cfg: DictConfig, data: Data):
    model = CONE(cfg.model, data)
    print(f"Model constructed:\n{model}")
    print(f"Total number of trainable parameters: {count_params(model):,}")

    # Load checkpoint
    if cfg.resume_model_ckpt:
        print(f"Loading moel checkpoint from {cfg.resume_model_ckpt}")
        ckpt = torch.load(cfg.resume_model_ckpt, map_location="cpu")
        model.load_state_dict(ckpt)

    return model


@contextmanager
def run_context(cfg: DictConfig):
    try:
        wandb_cfg = cfg.wandb
        if wandb_cfg.use:
            wandb.init(
                entity=wandb_cfg.entity,
                project=wandb_cfg.project,
                # group=cfg.name,
                name=cfg.fullname,
                config=cfg,
            )
        elif wandb_cfg.sweep:
            wandb.init()
        print(OmegaConf.to_yaml(cfg, sort_keys=True, resolve=True))
        yield
    finally:
        if wandb.run is not None:
            wandb.finish()


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    data = load_data(cfg)
    if cfg.dump_ctxt_nets:  # Dump proceseed contextualized networks
        dump_ctxt_nets(cfg.networks.name, cfg.context.name, data)
        exit()

    # Setup trainer and model
    model = setup_model(cfg, data)
    trainer = Trainer(cfg.trainer)

    # Send data and model to device
    device = get_device(cfg.device)
    data.to(device)
    model.to(device)

    with run_context(cfg):
        trainer.train(model, data)

    if torch.cuda.is_available():
        print(f"Peak CUDA memory usage: {torch.cuda.max_memory_allocated():,}")


if __name__ == "__main__":
    main()
