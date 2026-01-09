import json
import shutil
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Any

import accelerate
import diffusers
import torch
import wandb
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from accelerate.utils import set_seed
from hydra.utils import instantiate
from omegaconf import DictConfig
from omegaconf import ListConfig
from omegaconf import OmegaConf
from omegaconf.base import ContainerMetadata
from omegaconf.base import Metadata
from omegaconf.nodes import AnyNode

from . import helper
from .logger import AverageMeter

try:
    # avoid `weights_only` issue when loading state in accelerate
    torch.serialization.add_safe_globals([
        ListConfig,
        ContainerMetadata,
        Any, list,
        defaultdict,
        dict, int, float,
        AnyNode, Metadata
    ])
except:
    import warnings
    warnings.warn("cannot `add_safe_globals`")

# ------------------------------------------------------#
accelerate.checkpointing.load_model = helper.accelerate_load_model
# ------------------------------------------------------#


class Trainer:

    def __init__(self, hp: DictConfig):
        self.hp = hp
        self.project_dir = helper.get_run_dir()
        kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=3600))

        self.accelerator = Accelerator(
            gradient_accumulation_steps=hp.trainer.gradient_accumulation_steps,
            kwargs_handlers=[kwargs],
            mixed_precision=hp.trainer.mixed_precision,
            log_with=hp.trainer.log_with,
            project_dir=self.project_dir,
        )
        self.prepare()

    def prepare(self):
        set_seed(self.hp.seed + self.accelerator.process_index)
        hp = self.hp
        # ----------------------------Configure tracker ----------------------------#
        init_kwargs = {}
        if hp.trainer.log_with == "wandb":
            key = open(Path(__file__).parents[2].absolute() / ".key").read().strip()
            wandb.login(key=key)

            init_kwargs = {
                "wandb": {
                    "entity": hp.trainer.entity,
                    "name": hp.trainer.tracker_project_run_name,
                    "dir": self.project_dir,
                }
            }

        # get current configurations
        config = OmegaConf.to_container(hp, resolve=True)
        config = helper.flatten_dict(config)
        if self.hp.trainer.log_with == "tensorboard":
            # TB doesn't support list type yet
            config = {k: v for k, v in config.items() if type(v) != list}

        self.accelerator.init_trackers(
            self.hp.trainer.tracker_project_name,
            config=helper.flatten_dict(config),
            init_kwargs=init_kwargs,
        )
        # ---------------------------------------------------------------------------#

        model = instantiate(self.hp.pipeline)

        optimizer = instantiate(self.hp.trainer.optimizer, model.parameters())
        train_dl, eval_dl, test_dl = helper.collate_fn(self.hp)

        self.accelerator.print(f"#train={len(train_dl.dataset)}\t#valid={len(eval_dl.dataset)}")

        lr_scheduler = diffusers.optimization.get_scheduler(
            hp.trainer.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=hp.trainer.lr_warmup_steps,
            num_training_steps=hp.trainer.total_steps * self.accelerator.num_processes,
        )

        self.model, self.optimizer, self.lr_scheduler, \
            self.train_dl, self.eval_dl, self.test_dl, \
            = self.accelerator.prepare(
                model, optimizer, lr_scheduler,
                train_dl, eval_dl, test_dl,
            )

        self.best_score = dict()
        self.train_dl = helper.cycle_dataloader(self.train_dl)
        self.monitor = AverageMeter()
        self.step = 1

    def save_checkpoint(self):

        if not self.accelerator.is_main_process:
            return

        checkpoint = self.project_dir / "checkpoint"
        checkpoint_ = self.project_dir / "checkpoint_"

        if checkpoint_.exists():
            shutil.rmtree(checkpoint_)

        if checkpoint.exists() and checkpoint.is_dir():
            checkpoint.rename(checkpoint_)

        self.accelerator.save_state(checkpoint)
        self.accelerator.save({"step": self.step, **self.best_score}, checkpoint / "summary.pth")

    def try_load_checkpoint(self, checkpoint):
        if checkpoint.exists():
            try:
                self.accelerator.load_state(checkpoint, strict=False)
                ck_point = torch.load(checkpoint/"summary.pth", weights_only=False)
                self.step = ck_point.pop("step") + 1
                for k, v in ck_point.items():
                    self.best_score[k] = v
                del ck_point
                self.accelerator.print(f"Load checkpoint from {checkpoint}")
                return True

            except Exception as e:
                self.accelerator.print(f"Got error {e} when loading checkpoint from {checkpoint}")

        return False

    def load_checkpoint(self):
        checkpoint = self.project_dir / "checkpoint"
        checkpoint_ = self.project_dir / "checkpoint_"
        if not self.try_load_checkpoint(checkpoint):
            # breakpoint()
            if not self.try_load_checkpoint(checkpoint_):
                self.accelerator.print(f"Cannot load checkpoint. Training from scratch!")

    def train_one_step(self, model, **kargs):
        r"""Training step"""

        max_norm = self.hp.trainer.clip_grad_norm

        with self.accelerator.accumulate(model):
            output = model(next(self.train_dl), step=self.step)
            self.accelerator.backward(output["loss"].mean())

            self.monitor.update(output)

            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(model.parameters(), max_norm)

            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

            if self.step % self.hp.trainer.print_output_every == 0:
                lr = self.lr_scheduler.get_last_lr()[0]
                self.accelerator.print(f"[step={self.step},lr={lr:.2e}]\t" + str(self.monitor))
                self.write_log(self.monitor.average, f"train_{kargs['name']}")

                self.monitor.reset()

    @torch.no_grad()
    def valid_one_epoch(self, model, **kargs):
        r"""Validate the model."""
        model.eval()
        monitor = AverageMeter()

        for batch in self.eval_dl:
            output = model(batch, step=self.step)
            for k, v in output.items():
                value = self.accelerator.gather_for_metrics(v)
                monitor.update({k: value})

            if self.hp.get("debug", False):
                break

        self.accelerator.print(f"Step={self.step} [Valid-{kargs['name']}]\t" + str(monitor))
        self.write_log(monitor.average, f"valid_{kargs['name']}")

        self.valid_one_epoch_hook(model, **kargs)

        model.train()
        self.accelerator.wait_for_everyone()

    def valid_one_epoch_hook(self, model, **kargs):
        pass

    def train(self):
        r"""Train the model."""
        self.load_checkpoint()

        self.accelerator.print(helper.get_git_revision_hash())
        total, trainable, non_trainable = helper.numel(self.model)
        self.accelerator.print(f"# parameters\ttotal={total/1e6:.2f}M"
                               f"\ttrainable={trainable/1e6:.2f}M"
                               f"\tnon-trainable={non_trainable/1e6:.2f}M")

        self.model.train()
        for self.step in range(self.step, self.hp.trainer.total_steps):
            self.train_one_step(self.model, name="model")
            if self.step % self.hp.trainer.check_model_every == 0:
                self.valid_one_epoch(self.model, name="model")
                self.save_checkpoint()

        self.accelerator.wait_for_everyone()
        self.accelerator.end_training()

    def write_log(self, info: dict, key: str):

        if self.accelerator.is_main_process:
            self.accelerator.log({key + "/" + k: v for k, v in info.items()}, step=self.step)

            filename = self.project_dir / (key + ".json")
            info = {"step": self.step, **info}

            with open(filename, "a") as f:
                json.dump(info, f)
                f.write("\n")
