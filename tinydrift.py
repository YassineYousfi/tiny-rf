import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyDrift(nn.Module):
    def __init__(self, noise_dim, output_dim, hidden_size, n_layers, act, norm, temp=0.05):
        super().__init__()
        layers = [nn.Linear(noise_dim, hidden_size), norm(hidden_size), act()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_size, hidden_size), norm(hidden_size), act()])
        layers.append(nn.Linear(hidden_size, output_dim))
        self.net = nn.Sequential(*layers)
        self.noise_dim = noise_dim
        self.temp = temp
        print(f'number of parameters: {sum(p.numel() for p in self.parameters())}')

    def forward(self, z):
        return self.net(z)

    @staticmethod
    def compute_drift(gen, pos, temp=0.05):
        """Mean-shift drifting field with batch-normalized kernel."""
        targets = torch.cat([gen, pos], dim=0)
        G = gen.shape[0]

        dist = torch.cdist(gen, targets)
        dist[:, :G].fill_diagonal_(1e6)  # mask self
        kernel = (-dist / temp).exp()

        normalizer = (kernel.sum(dim=-1, keepdim=True) * kernel.sum(dim=-2, keepdim=True)).clamp_min(1e-12).sqrt()
        kernel = kernel / normalizer

        pos_coeff = kernel[:, G:] * kernel[:, :G].sum(dim=-1, keepdim=True)
        neg_coeff = kernel[:, :G] * kernel[:, G:].sum(dim=-1, keepdim=True)
        return pos_coeff @ targets[G:] - neg_coeff @ targets[:G]

    def forward_drift(self, x):
        """Drifting loss: MSE(gen, stopgrad(gen + V))."""
        z = torch.randn(x.size(0), self.noise_dim, device=x.device)
        gen = self(z)
        with torch.no_grad():
            V = self.compute_drift(gen, x, self.temp)
            target = (gen + V).detach()
        return F.mse_loss(gen, target)

    @torch.no_grad()
    def sample(self, z, num_steps=1):
        assert num_steps == 1
        return self(z)
