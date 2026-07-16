__all__ = ['BasicEncoderStatic', 'BasicEncoderStaticConfig', 'BasicEncoderStaticMLP', 'BasicEncoderStaticMLPConfig']

import torch
from torch import nn
import numpy as np

import dataclasses
from dataclasses import dataclass, field
from typing import List

from ..template_modules import EncoderStaticBase, EncoderStaticBaseConfig
from collections.abc import Iterable
from ..ts.basic_conv1d_modules.basic_conv1d import bn_drop_lin


class BasicEncoderStatic(EncoderStaticBase):
    def __init__(self, hparams_encoder_static, hparams_input_shape, target_dim=None):
        super().__init__(hparams_encoder_static, hparams_input_shape, target_dim)
        self.input_channels_cat = hparams_input_shape.static_dim_cat
        self.input_channels_cont = hparams_input_shape.static_dim
        assert(len(hparams_encoder_static.embedding_dims)==hparams_input_shape.static_dim_cat and len(hparams_encoder_static.vocab_sizes)==hparams_input_shape.static_dim_cat)
        self.embeddings = nn.ModuleList() if hparams_input_shape.static_dim_cat is not None else None
        for v,e in zip(hparams_encoder_static.vocab_sizes,hparams_encoder_static.embedding_dims):
            self.embeddings.append(nn.Embedding(v,e))
        self.input_dim = int(np.sum(hparams_encoder_static.embedding_dims) + hparams_input_shape.static_dim)
        self.input_channels = hparams_input_shape.static_dim + hparams_input_shape.static_dim_cat

    def embed(self, **kwargs):
        static = kwargs["static"] if "static" in kwargs.keys() else None
        static_cat = kwargs["static_cat"] if "static_cat" in kwargs.keys() else None

        res = []
        if(static_cat is not None):
            for i,e in enumerate(self.embeddings):
                res.append(e(static_cat[:,i].long()))
            if(static is not None and static_cat is not None):
                res = torch.cat([torch.cat(res,dim=1),static],dim=1)
            else:
                res = torch.cat(res,dim=1)
        else:
            res = static

        return res

    def forward(self, **kwargs):
        raise NotImplementedError

    def get_output_shape(self):
        raise NotImplementedError


@dataclass
class BasicEncoderStaticConfig(EncoderStaticBaseConfig):
    _target_:str = "clinical_ts.tabular.base.BasicEncoderStatic"
    embedding_dims:List[int] = field(default_factory=lambda: [])
    vocab_sizes:List[int] = field(default_factory=lambda: [])


# ── Aleatoric 버전 (mean + log_var head) ──────────────────────
class BasicEncoderStaticMLP(BasicEncoderStatic):
    def __init__(self, hparams_encoder_static, hparams_input_shape, target_dim=None):
        super().__init__(hparams_encoder_static, hparams_input_shape, target_dim)

        lin_ftrs = [self.input_dim] + list(hparams_encoder_static.lin_ftrs)
        if(target_dim is not None and lin_ftrs[-1] != target_dim):
            lin_ftrs.append(target_dim)

        ps = [hparams_encoder_static.dropout] if not isinstance(hparams_encoder_static.dropout, Iterable) else hparams_encoder_static.dropout
        if len(ps) == 1:
            ps = [ps[0]/2] * (len(lin_ftrs)-2) + ps
        actns = [nn.ReLU(inplace=True)] * (len(lin_ftrs)-2) + [None]

        # Shared layers (마지막 linear 제외)
        layers = []
        for ni,no,p,actn in zip(lin_ftrs[:-2], lin_ftrs[1:-1], ps[:-1], actns[:-1]):
            layers += bn_drop_lin(ni, no, hparams_encoder_static.batch_norm, p, actn, layer_norm=False)
        self.layers = nn.Sequential(*layers)

        last_hidden = lin_ftrs[-2]

        # Mean head (기존 output)
        self.mean_head    = nn.Linear(last_hidden, target_dim)
        # Log variance head (aleatoric uncertainty)
        self.log_var_head = nn.Linear(last_hidden, target_dim)
        nn.init.zeros_(self.log_var_head.weight)
        nn.init.zeros_(self.log_var_head.bias)
        self.output_shape = dataclasses.replace(hparams_input_shape)
        self.output_shape.static_dim = int(lin_ftrs[-1])
        self.output_shape.static_dim_cat = 0

    def forward(self, **kwargs):
        res = self.embed(**kwargs)
        hidden  = self.layers(res)
        mean    = self.mean_head(hidden)
        log_var = self.log_var_head(hidden)
        return {
            "static":  mean,
            "log_var": log_var
        }

    def get_output_shape(self):
        return self.output_shape


@dataclass
class BasicEncoderStaticMLPConfig(BasicEncoderStaticConfig):
    _target_:str = "clinical_ts.tabular.base.BasicEncoderStaticMLP"
    lin_ftrs:List[int] = field(default_factory=lambda: [512])
    dropout:float = 0.5
    batch_norm:bool = True