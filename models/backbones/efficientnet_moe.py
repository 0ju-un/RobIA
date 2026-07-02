
from typing import List

import torch
import torch.nn as nn

from .efficientnet import EfficientNetFeatures
from .efficientnet_blocks import DepthwiseSeparableConv, InvertedResidual, drop_path
from ..moe.utils import get_gate_network


class InvertedResidualMoE(InvertedResidual):
    def __init__(self, emb_dim=128, gate_type='linear', **kwargs):
        super(InvertedResidualMoE, self).__init__(**kwargs)
        gate_out_channels = kwargs.get('in_chs')
        if emb_dim == 'in_chs':
            emb_dim = kwargs.get('in_chs')

        self.embed_net = get_gate_network(gate_type, emb_dim)

        in_chs = kwargs['in_chs']
        out_chs = kwargs['out_chs']
        from timm.models.layers import make_divisible
        mid_chs = make_divisible(kwargs['in_chs'] * kwargs['exp_ratio'])

        self.gate1 = nn.Linear(emb_dim, mid_chs)
        self.gate2 = nn.Linear(emb_dim, mid_chs)
        self.gate3 = nn.Linear(emb_dim, out_chs)

        self.act = torch.sigmoid


    def forward(self, x, embedding=None):

        shortcut = x #([2, 96, 32, 56])

        gate_values = []
        embedding = self.embed_net(x)
        gate_value = self.act(self.gate1(embedding))

        gate_values.append(gate_value)
        # x = x * gate_value.unsqueeze(-1).unsqueeze(-1)

        # Point-wise expansion
        x = self.conv_pw(x)
        x = x * gate_value.unsqueeze(-1).unsqueeze(-1)
        x = self.bn1(x)
        x = self.act1(x)

        # Depth-wise convolution
        gate_value = self.act(self.gate2(embedding))
        gate_values.append(gate_value)

        x = self.conv_dw(x)
        x = x * gate_value.unsqueeze(-1).unsqueeze(-1)
        x = self.bn2(x)
        x = self.act2(x)

        # Squeeze-and-excitation
        x = self.se(x)

        # Point-wise linear projection
        gate_value = self.act(self.gate3(embedding))
        gate_values.append(gate_value)

        x = self.conv_pwl(x)
        x = x * gate_value.unsqueeze(-1).unsqueeze(-1)
        x = self.bn3(x)

        if self.has_residual:
            if self.drop_path_rate > 0.:
                x = drop_path(x, self.drop_path_rate, self.training)
            x += shortcut

        # x = x * gate_value.unsqueeze(-1).unsqueeze(-1)

        return x, gate_values

class EfficientNetFeaturesCustomMoE(EfficientNetFeatures):
    def __init__(self, block_args, dim=128, *args, **kwargs):
        patch_size_list = [16, 8, 4, 2, 2, 1]
        for i, b_list in enumerate(block_args):
            for j, b in enumerate(b_list):
                if 'attex-moe' in b['block_type']:
                    # b['gate_type'] = kwargs.get('gate_type')
                    b['gate_type'] = 'rowwise_self_attention_pool'
                    b['emb_dim']=kwargs.get('emb_dim', 'in_chs')
                    # b['activation'] = kwargs.get('activation')
                    b['activation'] = 'sigmoid'


        super(EfficientNetFeaturesCustomMoE, self).__init__(block_args, **kwargs)

    def forward(self, x, *args, **kwargs):
        embedding = self.embedding(x)

        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)

        out = []
        if self.feature_hooks is None:
            features = []
            if 0 in self._stage_out_idx:
                features.append(x)  # add stem out
            for i, b in enumerate(self.blocks):
                for j, _b in enumerate(b):
                    x = _b(x, embedding)
                    if isinstance(x, tuple):
                        x, o = x
                        out.append(o)
                if i + 1 in self._stage_out_idx:
                    features.append(x)
            return features, out
        else:
            self.blocks((x, embedding))
            out = self.feature_hooks.get_output(x.device)
            return list(out.values())
