import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim


def gumbel_softmax(logits, temperature=1.0, hard=False):
    gumbels = -torch.empty_like(logits).exponential_().log()  # Sample from Gumbel(0, 1)
    y = logits + gumbels
    probs = F.softmax(y / temperature, dim=-1)

    if hard:
        index = probs.max(dim=-1, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
        probs = (y_hard - probs).detach() + probs
    return probs


def gumbel_kl_divergence(logits):
    # Softmax to get the distribution q(y)
    q_y = torch.nn.functional.softmax(logits, dim=-1)

    # Compute log(q(y))
    log_q_y = torch.log(q_y + 1e-20)  # Adding a small constant to avoid log(0)

    # Compute KL divergence for binary values
    uniform_log_prob = torch.log(torch.tensor(0.5))
    kl_div = q_y * (log_q_y - uniform_log_prob) + (1 - q_y) * (torch.log(1 - q_y + 1e-20) - uniform_log_prob)

    return kl_div.sum(dim=-1).mean()


class GumbelVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(GumbelVAE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def reparameterize(self, logits, temperature):
        return gumbel_softmax(logits, temperature, hard=False)

    def forward(self, x, temperature):
        logits = self.encoder(x)
        z = self.reparameterize(logits, temperature)
        return self.decoder(z), logits

    def get_loss(self, x, recon_x, logits, temperature):
        reconstruction_loss = F.mse_loss(recon_x, x, reduction='mean')
        probs = F.softmax(logits, dim=-1)
        kl_div = gumbel_kl_divergence(probs)
        return reconstruction_loss, kl_div


# Training Loop
def train_gumbel_vae(model, data_loader, optimizer, epochs=200, initial_temperature=1.0, anneal_rate=0.003,
                     min_temperature=0.5):
    model.train()
    temperature = initial_temperature

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, (data, _) in enumerate(data_loader):
            optimizer.zero_grad()
            reconstructed, logits = model(data, temperature)

            rec_loss, kld = model.get_loss(data, reconstructed, logits, temperature)
            # print(rec_loss, kld)
            loss = rec_loss + kld
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Anneal the temperature
        temperature = max(temperature * np.exp(-anneal_rate * epoch), min_temperature)

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(data_loader)}')


if __name__ == "__main__":
    df = pd.read_csv('/media/nkatz/storage/EVENFLOW-DATA/DFKI/new-3-8-2023/DemoDataset_1Robot_clean.csv')

    X = df.iloc[:, :-1]

    scaler = MinMaxScaler()  # Might help when aiming for binary data
    X = scaler.fit_transform(X)

    train_tensors = torch.tensor(X, dtype=torch.float32)
    train_dataset = TensorDataset(train_tensors, train_tensors)

    batch_size = 32

    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = GumbelVAE(input_dim=10, hidden_dim=7, latent_dim=4)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    train_gumbel_vae(model, loader, optimizer)

    print('Training completed!')
