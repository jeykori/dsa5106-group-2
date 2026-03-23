# src/models/dora.py
# Minimal DoRA-style linear layer for quick experiments.
# This is a readable, small implementation to validate the decomposition idea.
# It uses per-output-row magnitude g and a direction matrix v, with a low-rank
# adapter A @ B applied to the direction.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoRALinear(nn.Module):
    """
    DoRA-style replacement for nn.Linear.
    - W (pretrained) is decomposed into g (out x 1) * v (out x in).
    - We apply a low-rank adapter to v: v <- v + A @ B
    - Trainable params: A, B (and optionally g).
    """
    def __init__(self, in_features, out_features, bias=True, rank=4, train_g=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.train_g = train_g

        # placeholder for original weight; will be initialized or loaded later
        self.W = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))

        # magnitude per output row (initialize from W norm)
        with torch.no_grad():
            g_init = self.W.norm(dim=1, keepdim=True) + 1e-8
        self.g = nn.Parameter(g_init, requires_grad=train_g)

        # direction matrix v (initialized from W / g)
        with torch.no_grad():
            v_init = self.W / self.g
        self.v = nn.Parameter(v_init.clone(), requires_grad=False)  # keep base direction frozen

        # low-rank adapter parameters (trainable)
        self.A = nn.Parameter(torch.zeros(out_features, rank))
        self.B = nn.Parameter(torch.zeros(rank, in_features))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x):
        # compute low-rank delta for direction
        delta_v = torch.matmul(self.A, self.B)  # shape: (out_features, in_features)
        v_hat = self.v + delta_v
        # normalize each output row to unit norm to keep direction semantics
        v_hat = F.normalize(v_hat, p=2, dim=1)
        # reconstruct weight: g (out x 1) * v_hat (out x in)
        W_hat = self.g * v_hat
        return F.linear(x, W_hat, self.bias)

    def load_from_linear(self, linear_module):
        """
        Initialize DoRA params from an existing nn.Linear module.
        Call this after creating DoRALinear to copy pretrained weights.
        """
        with torch.no_grad():
            w = linear_module.weight.data.clone()
            if linear_module.bias is not None and self.bias is not None:
                self.bias.data.copy_(linear_module.bias.data)
            # set W, g, v
            self.W.data.copy_(w)
            g_init = w.norm(dim=1, keepdim=True) + 1e-8
            self.g.data.copy_(g_init)
            self.v.data.copy_(w / g_init)

    def get_adapter_parameters(self):
        """Return the parameters that should be trained for DoRA (A, B, optionally g)."""
        params = [self.A, self.B]
        if self.train_g:
            params.append(self.g)
        return params
