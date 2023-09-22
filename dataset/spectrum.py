#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @file spectrum.py
# @author: wujiangu
# @date: 2023-06-01 21:11
# @description: spectrum dataset

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SpectrumDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        spectrum_length: int,
        class_names: list = ["A", "F", "G", "K", "M"],
        spectrum_suffix: str = ".csv",
    ) -> None:
        """spec and img dataset
        @param data_dir: data dir(include (specturm|photometric|label)/(*.csv|*.jpg|label.csv))
        @param spectrum_length: spectrum length
        @param class_names: class names
        @param spectrum_suffix: spectrum suffix
        """
        super().__init__()
        self.data_dir = data_dir
        self.spectrum_dir = os.path.join(data_dir, "spectrum")
        self.photometric_dir = os.path.join(data_dir, "photometric")
        self.label_dir = os.path.join(data_dir, "label")
        self.spectrum_length = spectrum_length
        self.class_names = class_names
        self.spectrum_suffix = spectrum_suffix

        # label(str) : [basename, label, ...]
        self.label = pd.read_csv(os.path.join(self.label_dir, "label.csv"))

        print("=" * 20)
        print(f'dataset: {data_dir.split("/")[-1]}')
        for label in self.class_names:
            label_count = len(self.label[self.label["label"] == label])
            print(f"{label}: {label_count}")
        print("=" * 20)

    def __len__(self) -> int:
        return len(self.label)

    def __getitem__(self, index: int) -> dict:
        """get item
        @param index: index
        @return ret: dict include spec, label
        """

        # read spectrum
        if self.spectrum_suffix == ".csv":
            spec = np.loadtxt(
                os.path.join(
                    self.spectrum_dir,
                    self.label["basename"].values[index] + self.spectrum_suffix,
                ),
                delimiter=",",
                dtype="float32",
            )
        elif self.spectrum_suffix == ".npy":
            spec = np.load(
                os.path.join(
                    self.spectrum_dir,
                    self.label["basename"].values[index] + self.spectrum_suffix,
                )
            )

        # read label
        label = self.label["label"].values[index]
        # label to long
        label = np.array(self.class_names.index(label), dtype="long")
        # to torch tensor
        label = torch.from_numpy(label)

        # spec to torch tensor
        spec = torch.from_numpy(spec)[:, 1]
        if len(spec) < self.spectrum_length:
            left_num = (self.spectrum_length - len(spec)) // 2
            right_num = self.spectrum_length - len(spec) - left_num
            spec = torch.cat(
                [
                    torch.zeros(left_num, dtype=torch.float32),
                    spec,
                    torch.zeros(right_num, dtype=torch.float32),
                ],
                dim=0,
            )

        spec = spec.unsqueeze(0)

        ret = {"spec": spec, "label": label}
        return ret
