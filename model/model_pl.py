#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @file model_pl.py
# @author: wujiangu
# @date: 2023-05-17 11:26
# @description: pytorch lightning model muti-scale partial conv

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torchmetrics
import wandb

from .muti_scale_pc import FasterNet


class BaselinePl(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = FasterNet(
            num_classes=cfg["num_classes"],
            spectrum_length=cfg["spectrum_length"],
            **cfg["pc"],
        )

        # train acc, val acc, test acc
        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=cfg["num_classes"]
        )
        self.val_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=cfg["num_classes"]
        )
        self.test_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=cfg["num_classes"]
        )

        self.loss_fn = torch.nn.CrossEntropyLoss()

        # test recall, precision, f1
        self.test_recall = torchmetrics.Recall(
            task="multiclass", num_classes=cfg["num_classes"], average="macro"
        )
        self.test_precision = torchmetrics.Precision(
            task="multiclass", num_classes=cfg["num_classes"], average="macro"
        )
        self.test_f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=cfg["num_classes"], average="macro"
        )

        # a pandas to store test label and pred
        self.test_label_pred = pd.DataFrame(data=None, columns=["label", "pred"])
        # best val acc
        self.best_val_acc = 0.0

    def training_step(self, batch, _):
        spec = batch["spec"]
        label = batch["label"]
        pred = self.model(spec)
        loss = self.loss_fn(pred, label)
        self.train_acc(pred, label)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=spec.size(0),
        )
        return loss

    def on_train_epoch_end(self):
        self.log("train_acc", self.train_acc.compute(), on_epoch=True, prog_bar=True)
        self.train_acc.reset()

    def validation_step(self, batch, _):
        spec = batch["spec"]
        label = batch["label"]
        pred = self.model(spec)
        loss = self.loss_fn(pred, label)
        self.val_acc(pred, label)
        self.log(
            "val_loss", loss, on_epoch=True, prog_bar=True, batch_size=spec.size(0)
        )
        return loss

    def on_validation_epoch_end(self):
        # if sanity check, return
        if not self.trainer.sanity_checking:
            val_acc = self.val_acc.compute()
            self.log("val_acc", val_acc, on_epoch=True, prog_bar=True)
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
            self.log(
                "best_val_acc",
                self.best_val_acc,
                prog_bar=True,
            )

        self.val_acc.reset()

    def test_step(self, batch, _):
        spec = batch["spec"]
        label = batch["label"]
        pred = self.model(spec)
        loss = self.loss_fn(pred, label)
        self.test_acc(pred, label)
        self.test_recall(pred, label)
        self.test_precision(pred, label)
        self.test_f1(pred, label)

        # store test label and pred
        self.test_label_pred = pd.concat(
            [
                self.test_label_pred,
                pd.DataFrame(
                    data=np.array(
                        [label.cpu().numpy(), pred.argmax(dim=1).cpu().numpy()]
                    ).T,
                    columns=["label", "pred"],
                ),
            ],
            axis=0,
        )

        self.log(
            "test_loss", loss, on_epoch=True, prog_bar=True, batch_size=spec.size(0)
        )
        return loss

    def on_test_epoch_end(self):
        self.log(
            "test_acc",
            self.test_acc.compute(),
        )
        self.log(
            "test_recall",
            self.test_recall.compute(),
        )
        self.log(
            "test_precision",
            self.test_precision.compute(),
        )
        self.log(
            "test_f1",
            self.test_f1.compute(),
        )
        self.test_acc.reset()
        self.test_recall.reset()
        self.test_precision.reset()
        self.test_f1.reset()

        if self.cfg["log"]:
            _cm = wandb.plot.confusion_matrix(
                probs=None,
                y_true=self.test_label_pred["label"].to_numpy(),
                preds=self.test_label_pred["pred"].to_numpy(),
                class_names=self.cfg["class_names"],
                title=f"test_confusion_matrix",
            )
            wandb.log(
                {
                    f"test_confusion_matrix": _cm,
                }
            )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg["lr"])
        # optimizer = torch.optim.SGD(self.parameters(), lr=self.cfg["lr"], momentum=0.9, weight_decay=1e-4)
        # optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg["lr"], weight_decay=1e-4)
        # optimizer = torch.optim.RMSprop(self.parameters(), lr=self.cfg["lr"], weight_decay=1e-4)
        # optimizer = torch.optim.Adadelta(self.parameters(), lr=self.cfg["lr"], weight_decay=1e-4)
        # optimizer = torch.optim.Adagrad(self.parameters(), lr=self.cfg["lr"], weight_decay=1e-4)
        # optimizer = torch.optim.Adamax(self.parameters(), lr=self.cfg["lr"], weight_decay=1e-4)
        lr_scheduler = {
            "scheduler": optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=self.cfg["T_0"],
                T_mult=self.cfg["T_mult"],
                eta_min=self.cfg["eta_min"],
            ),
            "name": "lr",
        }

        return [optimizer], [lr_scheduler]


if __name__ == "__main__":
    pass
