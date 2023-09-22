#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @file pc.py
# @author: wujiangu
# @date: 2023-05-29 19:55
# @description: muti-scale partial conv for spectrum

import math
from typing import List

import torch
import torch.nn as nn
from timm.models.layers import DropPath


# ECA attention
class ECA(nn.Module):
    """ECA attention"""

    def __init__(self, channel: int):
        super(ECA, self).__init__()

        k_size = math.ceil(math.log(channel, 2) / 2 + 0.5)

        k_size = int(k_size)
        if k_size % 2 == 0:
            k_size += 1

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(
            1,
            1,
            kernel_size=k_size,
            padding=(k_size - 1) // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y = y.transpose(-1, -2)
        y = self.conv(y)
        y = self.sigmoid(y)
        # print(y[0])
        # with open("eca.txt", "a") as f:
        #     out = torch.mean(y, dim=0).squeeze().squeeze().cpu().numpy()
        #     for i in out:
        #         f.write(str(i) + ",")
        #     f.write("\n")
        # print(y.shape)
        y = y.transpose(-1, -2)
        return x * y


class PartialConv1D(nn.Module):
    """Partial Convolution for Spectrum"""

    def __init__(
        self,
        channel: int,
        n_div: int,
        pc_conv_size: int = 32,
        pc_conv_size_scale: float = 1.5,
    ):
        super(PartialConv1D, self).__init__()

        self.channel_div1 = channel // n_div
        self.channel_div2 = self.channel_div1
        self.channel_div3 = self.channel_div1
        self.conv1 = torch.nn.Conv1d(
            in_channels=self.channel_div1,
            out_channels=self.channel_div1,
            kernel_size=pc_conv_size,
            stride=1,
            padding="same",
            bias=False,
        )
        self.conv2 = torch.nn.Conv1d(
            in_channels=self.channel_div2,
            out_channels=self.channel_div2,
            kernel_size=int(pc_conv_size * pc_conv_size_scale),
            stride=1,
            padding="same",
            bias=False,
        )
        self.conv3 = torch.nn.Conv1d(
            in_channels=self.channel_div2,
            out_channels=self.channel_div2,
            kernel_size=int(pc_conv_size * pc_conv_size_scale * pc_conv_size_scale),
            stride=1,
            padding="same",
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_div1 = x[:, : self.channel_div1, :]
        x_div2 = x[:, self.channel_div1 : self.channel_div1 * 2, :]
        x_div3 = x[:, self.channel_div1 * 2 : self.channel_div1 * 3, :]
        x_rest = x[:, self.channel_div1 * 3 :, :]

        x_div1 = self.conv1(x_div1)
        x_div2 = self.conv2(x_div2)
        x_div3 = self.conv3(x_div3)

        x = torch.cat([x_div1, x_div2, x_div3, x_rest], dim=1)

        return x


class MLPBlock(nn.Module):
    """MLP Block"""

    def __init__(
        self,
        channel: int,
        n_div: int,
        mlp_ratio: float,
        drop_path_rate: float,
        act_layer: torch.nn.Module,
        norm_layer: torch.nn.Module,
        pc_conv_size: int = 32,
        pc_conv_size_scale: float = 1.0,
        attention: str = "",
        stage_idx: int = 0,
        spectrum_length: int = 1024,
    ):
        super(MLPBlock, self).__init__()

        mlp_hidden_channel = int(channel * mlp_ratio)

        mlp_layers: List[torch.nn.Module] = [
            ECA(channel) if attention == "ECA_F" else nn.Identity(),
            nn.Conv1d(channel, mlp_hidden_channel, 1, bias=False),
            norm_layer(mlp_hidden_channel),
            act_layer(inplace=True) if type(act_layer) == nn.ReLU else act_layer(),
            nn.Conv1d(mlp_hidden_channel, channel, 1, bias=False),
            ECA(channel) if attention == "ECA_B" else nn.Identity(),
        ]

        self.mlp = nn.Sequential(*mlp_layers)

        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        )

        self.spatial_mixing = PartialConv1D(
            channel, n_div, pc_conv_size, pc_conv_size_scale
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(self.mlp(x))

        return x


class BasicStage(nn.Module):
    """Basic Stage"""

    def __init__(
        self,
        channel: int,
        n_div: int,
        mlp_ratio: float,
        drop_path_rate: list[float],
        act_layer: torch.nn.Module,
        norm_layer: torch.nn.Module,
        depth: int,
        pc_conv_size: int = 32,
        pc_conv_size_scale: float = 1.0,
        attention: str = "",
        stage_idx: int = 0,
        spectrum_length: int = 2048,
    ):
        super(BasicStage, self).__init__()

        blocks: List[torch.nn.Module] = [
            MLPBlock(
                channel=channel,
                n_div=n_div,
                mlp_ratio=mlp_ratio,
                drop_path_rate=drop_path_rate[i],
                act_layer=act_layer,
                norm_layer=norm_layer,
                pc_conv_size=pc_conv_size,
                pc_conv_size_scale=pc_conv_size_scale,
                attention=attention,
                stage_idx=stage_idx,
                spectrum_length=spectrum_length,
            )
            for i in range(depth)
        ]

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        return x


class PatchEmbed(nn.Module):
    """Patch Embedding"""

    def __init__(
        self,
        conv_size: int,
        conv_stride: int,
        in_channel: int,
        out_channel: int,
        norm_layer: torch.nn.Module,
    ):
        super(PatchEmbed, self).__init__()

        self.proj = nn.Conv1d(
            in_channel,
            out_channel,
            kernel_size=conv_size,
            stride=conv_stride,
            bias=False,
        )

        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.norm(x)
        return x


class PatchMerging(nn.Module):
    """Patch Merging"""

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        conv_size: int,
        conv_stride: int,
        norm_layer: torch.nn.Module,
    ):
        super(PatchMerging, self).__init__()

        self.reduction = nn.Conv1d(
            in_channel,
            out_channel,
            kernel_size=conv_size,
            stride=conv_stride,
            bias=False,
        )

        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.reduction(x)
        x = self.norm(x)
        return x


class MSPCNet(nn.Module):
    """MPSCNet"""

    def __init__(
        self,
        in_channel: int = 1,
        num_classes: int = 5,
        embed_dim: int = 128,
        depth: list[int] = [1, 2, 8, 2],
        depth_scale: float = 1.0,
        mp_ratio: float = 2.0,
        n_div: int = 4,
        patch_conv_size: int = 64,
        patch_conv_stride: int = 2,
        pc_conv_size: int = 32,
        pc_conv_size_scale: float = 1.5,
        merge_conv_size: int = 2,
        merge_conv_stride: int = 2,
        head_dim: int = 128,
        drop_path_rate: float = 0.3,
        norm_layer: str = "BN",
        act_layer: str = "RELU",
        attention: str = "",
        spectrum_length: int = 1024,
    ):
        super(MSPCNet, self).__init__()

        depth = [int(x * depth_scale) for x in depth]
        if norm_layer == "BN":
            norm_layer = nn.BatchNorm1d
        elif norm_layer == "LN":
            norm_layer = nn.LayerNorm

        if act_layer == "RELU":
            act_layer = nn.ReLU
        elif act_layer == "GELU":
            act_layer = nn.GELU

        self.num_stage = len(depth)
        self.patch_embed = PatchEmbed(
            conv_size=patch_conv_size,
            conv_stride=patch_conv_stride,
            in_channel=in_channel,
            out_channel=embed_dim,
            norm_layer=norm_layer,
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]

        stages_list = []
        for i in range(self.num_stage):
            stage = BasicStage(
                channel=embed_dim,
                n_div=n_div,
                mlp_ratio=mp_ratio,
                drop_path_rate=dpr[sum(depth[:i]) : sum(depth[: i + 1])],
                act_layer=act_layer,
                norm_layer=norm_layer,
                depth=depth[i],
                pc_conv_size=pc_conv_size,
                pc_conv_size_scale=pc_conv_size_scale,
                attention=attention,
                stage_idx=i,
                spectrum_length=spectrum_length,
            )

            stages_list.append(stage)

            if i != self.num_stage - 1:
                stage = PatchMerging(
                    in_channel=embed_dim,
                    out_channel=embed_dim,
                    conv_size=merge_conv_size,
                    conv_stride=merge_conv_stride,
                    norm_layer=norm_layer,
                )

                stages_list.append(stage)

        self.stages = nn.Sequential(*stages_list)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(embed_dim, head_dim, 1, bias=False),
            act_layer(),
            nn.Flatten(),
            nn.Linear(head_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self.stages(x)
        x = self.head(x)
        return x


if __name__ == "__main__":
    model = MSPCNet()
    print(model)

    x = torch.randn(2, 1, 3584)
    import torchinfo

    torchinfo.summary(model, input_size=(2, 1, 3584), depth=6)
    y = model(x)
    print(y.shape)
