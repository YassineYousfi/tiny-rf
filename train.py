import sys
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import dino_dataset
from tqdm.auto import tqdm

mode = sys.argv[1] if len(sys.argv) > 1 else 'rf'
assert mode in ('rf', 'drift'), f"usage: python train.py [rf|drift]"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
output_dim = 2
n_visualize_samples = 1_000
act = torch.nn.SiLU

if mode == 'rf':
    from tinyrf import TinyRF
    hidden_size, n_layers, batch_size = 128, 8, 64
    epochs, lr, lr_decay = 50, 1e-3, 1.05
    sample_steps = 15
    model = TinyRF(output_dim, output_dim, hidden_size, n_layers, act, torch.nn.LayerNorm).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    compute_loss = lambda batch: model.forward_rf(batch)
    sample_noise_dim = output_dim
else:
    from tinydrift import TinyDrift
    noise_dim, hidden_size, n_layers, batch_size = 32, 256, 4, 2048
    epochs, lr, lr_decay = 260, 1e-3, 1.0
    sample_steps = 1
    temp = 0.2
    model = TinyDrift(noise_dim, output_dim, hidden_size, n_layers, act, torch.nn.LayerNorm, temp=temp).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    compute_loss = lambda batch: model.forward_drift(batch)
    sample_noise_dim = noise_dim

dataset = dino_dataset(n=16_000)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

pbar = tqdm(total=len(dataloader))
model.train()
for epoch in range(epochs):
    epoch_loss = []
    pbar.reset(); pbar.set_description(f"epoch={epoch}")
    for batch in dataloader:
        loss = compute_loss(batch[0].to(device))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss.append(loss.detach().item())
        pbar.update(); pbar.set_postfix(loss=sum(epoch_loss) / len(epoch_loss))
    for g in optimizer.param_groups: g['lr'] /= lr_decay
pbar.close()

model.eval()
noise = torch.randn(n_visualize_samples, sample_noise_dim, device=device)
model_samples = model.sample(noise, num_steps=sample_steps).cpu().numpy()
plt.scatter(model_samples[:, 0], model_samples[:, 1], s=3, label=mode)
plt.scatter(dataset.tensors[0][:n_visualize_samples, 0], dataset.tensors[0][:n_visualize_samples, 1], s=3, label='data')
plt.legend()
plt.show()
