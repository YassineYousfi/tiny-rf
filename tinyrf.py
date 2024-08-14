import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class ModulatedLinAct(nn.Module):
    def __init__(self, input_ch, output_ch, modulation_ch, act, norm):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(input_ch, output_ch),
            act()
        )
        self.mod = nn.Sequential(
            act(),
            nn.Linear(modulation_ch, 2*input_ch)
        )
        self.norm = norm(input_ch)

    def forward(self, x, t):
        x = self.norm(x)
        shift, scale = self.mod(t).split(x.size(1), dim=1)
        return self.ff(x * (1 + scale) + shift)

class TinyRF(nn.Module):
    def __init__(self, input_ch, output_ch, hidden_size, n_layers, act, norm):
        super().__init__()
        self.timestep_embed = nn.Linear(1, hidden_size)
        self.input_embed = nn.Linear(input_ch, hidden_size)
        self.mlp = nn.ModuleList(
            [ModulatedLinAct(hidden_size, hidden_size, hidden_size, act, norm) for _ in range(n_layers)]
        )
        self.last_layer = nn.Linear(hidden_size, output_ch)
        self.initialize_weights()
        print(f'number of parameters: {sum(p.numel() for p in self.parameters())}')

    # is this necessary?
    def initialize_weights(self):
        for layer in self.mlp:
            nn.init.constant_(layer.mod[-1].weight, 0)
            nn.init.constant_(layer.mod[-1].bias, 0)

        nn.init.constant_(self.last_layer.weight, 0)
        nn.init.constant_(self.last_layer.bias, 0)

    def forward(self, x, t):
        t = self.timestep_embed(t)
        x = self.input_embed(x)
        for layer in self.mlp:
            x = x + layer(x, t)
        return self.last_layer(x)

    def forward_rf(self, x):
        b = x.size(0)
        t = torch.sigmoid(torch.randn((b,)).to(x.device)).view([b, 1])
        z1 = torch.randn_like(x)
        zt = (1 - t) * x + t * z1
        v_pred = self(zt, t)
        v_true = x - z1
        return F.mse_loss(v_pred, v_true)

    @torch.no_grad()
    def sample(self, z, num_steps):
        b = z.size(0)
        dt = torch.tensor([1.0 / num_steps] * b).to(z.device).view([b, 1])
        for i in range(num_steps, 0, -1):
            t = torch.tensor([i / num_steps] * b).to(z.device).view([b, 1])
            v = self(z, t)
            z = z + dt * v
        return z
