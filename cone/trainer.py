import os
from pathlib import Path
from time import perf_counter
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor
from torch import optim
from tqdm import tqdm, trange

from cone.config import BASE_CONDITION_NAME
from cone.utils import maybe_log_wandb, sanitize_name


class Trainer:

    def __init__(self, trainer_cfg: DictConfig):
        self.optimizer = trainer_cfg.optimizer
        self.optimizer_kwargs = trainer_cfg.optimizer_kwargs or {}
        self.lr = trainer_cfg.lr
        self.weight_decay = trainer_cfg.weight_decay
        self.max_epochs = trainer_cfg.max_epochs
        self.clip_grad_norm = trainer_cfg.clip_grad_norm
        self.track_grad_norm = trainer_cfg.track_grad_norm
        self.ctxt_lambda = trainer_cfg.ctxt_lambda
        self.dump_cfg = trainer_cfg.dump
        self.reset_epoch_stats()

    def reset_epoch_stats(self, **kwargs):
        self._epoch_stats = kwargs

    def train(self, model, data):
        if (optim_cls := getattr(optim, self.optimizer, None)) is None:
            raise ValueError(f"Unkown optimizer {self.optimizer!r}")
        optimizer = optim_cls(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            **self.optimizer_kwargs,
        )

        for epoch in (pbar := trange(self.max_epochs)):
            self.reset_epoch_stats(epoch=epoch)
            tic = perf_counter()

            z, loss = self.train_epoch(model, data, optimizer)

            if self.is_ckpt_epoch():
                self.dump_ckpt(model, data, epoch, z)

            self._epoch_stats["time_epoch"] = perf_counter() - tic
            maybe_log_wandb(self._epoch_stats)

            pbar.set_description(f"Epoch {epoch:>4}, loss: {loss:.5f}")

    def train_epoch(self, model, data, optimizer) -> Tuple[Tensor, float]:
        model.train()
        tot_loss = 0
        step_stats = {}

        for ctxt, pos_rw, neg_rw in (pbar := tqdm(model.rw_loader)):
            scale = 1 if ctxt is None else self.ctxt_lambda
            z = model(data.x, data.edge_index, data.edge_weight, ctxt=ctxt)
            loss = model.compute_loss(z, pos_rw, neg_rw, ctxt=ctxt)
            scaled_loss = scale * loss

            optimizer.zero_grad()
            scaled_loss.backward()
            if self.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad_norm)
            optimizer.step()

            # Record stats
            if self.track_grad_norm:
                step_stats["train/grad_norm_step"] = get_grad_norms(model).mean().item()
            step_stats["train/loss_step"] = loss.item()
            maybe_log_wandb(step_stats)

            tot_loss += step_stats["train/loss_step"]
            step_stats.clear()

            name = ctxt or BASE_CONDITION_NAME
            pbar.set_description(f"loss: {loss:>9.5f}, ctxt: {name:<20}")

        self._epoch_stats["train/loss_epoch"] = tot_loss / len(model.rw_loader)

        # Prepare dumps
        embeddings = {}
        if self.is_ckpt_epoch():
            for ctxt in [None] + list(model.cond_embs):
                embeddings[ctxt or BASE_CONDITION_NAME] = model(
                    data.x,
                    data.edge_index,
                    data.edge_weight,
                    ctxt=ctxt,
                ).detach().to("cpu", non_blocking=True)

        return embeddings, self._epoch_stats["train/loss_epoch"]

    def is_ckpt_epoch(self) -> bool:
        epoch = self._epoch_stats["epoch"]
        return (
            self.dump_cfg.enable
            and (epoch > 0)
            and (epoch % self.dump_cfg.dump_interval == 0)
        )

    @torch.no_grad()
    def dump_ckpt(self, model, data, epoch, z):
        out_path = Path(self.dump_cfg.path)
        os.makedirs(out_path, exist_ok=True)

        # Dump node ids
        node_ids_dump_path = out_path / "node_ids.npy"
        if not os.path.isfile(node_ids_dump_path):
            np.save(out_path / "node_ids.npy", data.node_ids)

        # Dump embeddings
        if self.dump_cfg.dump_embeddings:
            emd_out_path = out_path / "embeddings"
            os.makedirs(emd_out_path, exist_ok=True)
            os.makedirs(emd_out_path / "latest", exist_ok=True)

            for name, z_ in z.items():
                name = sanitize_name(name)
                z_array = z_.numpy()
                np.save(emd_out_path / "latest" / f"{name}.latest.npy", z_array)
                np.save(emd_out_path / f"{name}.{epoch}.npy", z_array)

        # Dump model
        if self.dump_cfg.dump_model:
            mdl_out_path = out_path / "models"
            os.makedirs(mdl_out_path, exist_ok=True)

            torch.save(model.state_dict(), mdl_out_path / "latest.pt")
            torch.save(model.state_dict(), mdl_out_path / f"{epoch}.pt")


@torch.no_grad()
def get_grad_norms(model):
    p_norm_list = []
    for p in model.parameters():
        if p.grad is not None:
            p_norm = p.grad.data.norm(2)
            p_norm_list.append(p_norm)
    return torch.tensor(p_norm_list)
