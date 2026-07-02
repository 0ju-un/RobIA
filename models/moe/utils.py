import torch
import torch.nn as nn
from torch.nn import functional as F
import math

import wandb


#"""
class MoELayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        
    def forward(self, x, gate_values):
        gate_values = gate_values.to(x.dtype)
        batch_size = x.size(0)
        gate_values = gate_values.view(batch_size, -1, 1, 1)
        output = self.conv_layer(x)
        output = output * gate_values
        return output


class MoELayer3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=False, stride=1, padding=0, dilation=1, deconv=False):
        super().__init__()
        if deconv:
            self.conv_layer = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, bias=bias, stride=stride, padding=padding, dilation=dilation)
        else:
            self.conv_layer = nn.Conv3d(in_channels, out_channels, kernel_size, bias=bias, stride=stride, padding=padding, dilation=dilation)

    def forward(self, x, gate_values):
        gate_values = gate_values.to(x.dtype)
        batch_size = x.size(0)
        gate_values = gate_values.view(batch_size, -1, 1, 1,1)
        output = self.conv(x)
        output = output * gate_values
        return output
    

class RowWiseSelfAttentionPooling(nn.Module):
    def __init__(self, input_dim, num_experts):
        super().__init__()
        self.q_proj = nn.Linear(input_dim, input_dim)
        self.k_proj = nn.Linear(input_dim, input_dim)
        self.v_proj = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        """
        x: Tensor of shape (B, C, H, W)
        returns: Tensor of shape (B, C, H, W)
        """
        if x.dim() == 5:
            B, _, C, H, W = x.shape # x.shape (B, C, D,H,W)
            x = x.mean(dim=1)
        else:
            B, C, H, W = x.shape


        # Reshape to treat each row as a sequence: (B*H, W, C)
        x = x.permute(0, 2, 3, 1).reshape(B * H, W, C)

        # Self-attention
        Q = self.q_proj(x)   # (B*H, W, C)
        K = self.k_proj(x)
        V = self.v_proj(x)

        d_k = Q.shape[-1] # d_k == C
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / (d_k ** 0.5)  # (B*H, W, W)
        attn_weights = F.softmax(attn_scores, dim=-1)                # (B*H, W, W)

        attn_out = torch.bmm(attn_weights, V)  # (B*H, W, C)

        # Reshape back to (B, C, H, W)
        out = attn_out.reshape(B, H, W, C).permute(0, 3, 1, 2)  # B × C × H × W
        pooled = out.mean(dim=[2, 3])
        a=1
        return pooled

GATE_LUT = {
    'rowwise_self_attention_pool': RowWiseSelfAttentionPooling,
}

def get_gate_network(gate_type='linear', emb_dim=128, out_channels=192, patch_size=None):
    if gate_type not in GATE_LUT:
        raise ValueError(f'Unknown gate type: {gate_type}.')
    return GATE_LUT[gate_type](emb_dim, out_channels)


        