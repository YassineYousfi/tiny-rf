import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tinyrf import TinyRF
from dataset import dino_dataset
from tqdm.auto import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_ch = output_ch = 2
hidden_size = 128
n_layers = 8
batch_size = 64
epochs = 50
n_visualize_samples = 1_000
num_steps = 15
lr = 1e-3
lr_decay = 1.05
act = torch.nn.SiLU
norm = torch.nn.LayerNorm
model = TinyRF(input_ch, output_ch, hidden_size, n_layers, act, norm).train(mode=False).to(device=device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

dataset = dino_dataset(n=16_000)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

pbar = tqdm(total=len(dataloader))
model = model.train(mode=True)
for epoch in range(epochs):
    epoch_loss = []
    pbar.reset(); pbar.set_description(f"epoch={epoch}, lr={optimizer.param_groups[0]['lr']:.2e}")
    for batch in dataloader:
        batch = batch[0].to(device=device)
        loss = model.forward_rf(batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss.append(loss.detach().item())
        pbar.update(); pbar.set_postfix(loss=sum(epoch_loss) / len(epoch_loss))
    for g in optimizer.param_groups: g['lr'] /= lr_decay
pbar.close()

model = model.train(mode=False)
noise = torch.randn((n_visualize_samples, 2), device=device)
model_samples = model.sample(noise, num_steps=num_steps).cpu().numpy()
plt.scatter(model_samples[:,0], model_samples[:,1], s=3)
plt.scatter(dataset.tensors[0][:n_visualize_samples, 0], dataset.tensors[0][:n_visualize_samples, 1], s=3)
plt.show()
