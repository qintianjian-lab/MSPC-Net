#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @file train.py
# @author: wujiangu
# @date: 2023-05-17 13:48
# @description: train

import os

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from torch.utils.data import DataLoader

from cfg.cfg import cfg
from dataset.spectrum import SpectrumDataset
from model.model_pl import BaselinePl
from utils.tools import arg_parser, get_free_gpu_index, seed_every_thing

torch.set_float32_matmul_precision("high")


def main():
    # 1. seed everything and set gpu
    seed_every_thing(cfg["seed"])
    if cfg["device"] == "auto":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(
            get_free_gpu_index(cfg["device_list"]))
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg["device"]

    # 2. load data
    train_dataset = SpectrumDataset(
        data_dir=os.path.join(cfg["data_dir"], "train"),
        class_names=cfg["class_names"],
        spectrum_length=cfg["spectrum_length"],
    )
    val_dataset = SpectrumDataset(
        data_dir=os.path.join(cfg["data_dir"], "val"),
        class_names=cfg["class_names"],
        spectrum_length=cfg["spectrum_length"],
    )
    test_dataset = SpectrumDataset(
        data_dir=os.path.join(cfg["data_dir"], "test"),
        class_names=cfg["class_names"],
        spectrum_length=cfg["spectrum_length"],
    )

    # 3. load dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )

    # 4. model
    baseline_pl = BaselinePl(cfg)

    # 5. load trainer
    # 5.1 set logger
    if cfg["log"]:
        loggers = []
        log_path = cfg["log_path"]
        project = cfg["project"]
        sweep = cfg["sweep"]

        t_save_dir = os.path.join(log_path, project)
        t_logger = TensorBoardLogger(save_dir=t_save_dir, name=sweep)
        t_logger.log_hyperparams(cfg)
        t_logger.save()
        version = f"version_{t_logger.version}"

        w_save_dir = os.path.join(t_save_dir, sweep, version)

        # get device ip and user name
        device_ip = os.popen("hostname -I").read().strip()
        user_name = os.popen("whoami").read().strip()

        print("=" * 50, version, "=" * 50, sep="\n")
        w_logger = WandbLogger(
            project=cfg["project"],
            save_dir=w_save_dir,
            name=f"{sweep}_{version}_{user_name}@{device_ip}",
        )
        loggers.append(w_logger)
        loggers.append(t_logger)
        # 3.2 set callbacks
        checkpoint_callback = ModelCheckpoint(
            monitor="val_acc",
            dirpath=os.path.join(w_save_dir, "checkpoints"),
            filename="{epoch:03d}-{val_acc:.4f}",
            save_top_k=3,
            mode="max",
        )
        lr_monitor = LearningRateMonitor(logging_interval="step")

        early_stop_callback = EarlyStopping(
            monitor="val_acc",
            patience=cfg["patience"],
            verbose=True,
            mode="max",
            min_delta=cfg["min_delta"],
        )

        callbacks = [
            checkpoint_callback,
            lr_monitor,
            early_stop_callback,
            ModelSummary(-1),
        ]

    # 5.2 set trainer
    trainer = Trainer(
        max_epochs=cfg["epochs"],
        logger=loggers if cfg["log"] else None,
        callbacks=callbacks if cfg["log"] else [ModelSummary(-1)],
        log_every_n_steps=1,
        fast_dev_run=cfg["debug"],
        precision=cfg["precision"],
    )

    trainer.fit(baseline_pl, train_dataloader, val_dataloader)

    # 6. test
    if cfg["test"] != 0:
        trainer.test(
            baseline_pl,
            test_dataloader,
            ckpt_path=None
            if not cfg["log"] else checkpoint_callback.best_model_path,
        )


if __name__ == "__main__":
    arg_parser(cfg)
    main()
