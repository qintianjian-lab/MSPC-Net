#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @file cfg.py
# @author: wujiangu
# @date: 2023-05-17 13:32
# @description: config file

cfg = {
    # train params
    "device": "auto",
    "device_list": [0, 1, 2, 3],
    "seed": 42,
    "debug": False,
    "log": False,
    "test": True,
    "precision": "16-mixed",
    "num_workers": 16,
    "patience": 25,
    "min_delta": 0.001,
    # model params
    "in_channel": 1,
    # dataset params
    "num_classes": 5,
    "data_dir": "your data dir",
    "class_names": "your class names",
    "spectrum_length": 3584,
    # log params
    "project": "your project name",
    "sweep": "your sweep name",
    "log_path": "your log path",
    # hyper params
    "epochs": 100,
    "lr": 1e-4,
    "eta_min": 1e-8,
    "T_0": 40,
    "T_mult": 2,
    "batch_size": 32,
    "pc": {
        "in_channel": 1,
        "embed_dim": 128,
        "depth": [1, 1, 2, 1],
        "depth_scale": 2.0,
        "mp_ratio": 1.0,
        "n_div": 8,
        "patch_conv_size": 3,
        "patch_conv_stride": 1,
        "pc_conv_size": 5,
        "pc_conv_size_scale": 9.0,
        "merge_conv_size": 3,
        "merge_conv_stride": 2,
        "head_dim": 1024,
        "drop_path_rate": 0.2,
        "norm_layer": "BN",
        "act_layer": "GELU",
        "attention": "ECA_F",
    },
}
