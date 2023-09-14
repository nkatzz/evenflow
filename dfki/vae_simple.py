import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

"""
Trains a simple VAE on the DFKI dataset.
"""

# 1. Load and preprocess data
data = pd.read_csv('/media/nkatz/storage/EVENFLOW-DATA/DFKI/new-3-8-2023/DemoDataset_1Robot.csv')
scaler = StandardScaler()
data_relevant_scaled = scaler.fit_transform(data[['px', 'py', 'pz', 'ox', 'oy', 'oz', 'ow', 'vx', 'vy', 'wz']])
data_tensor = torch.tensor(data_relevant_scaled, dtype=torch.float32)


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()

        # Encoder layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder layers
        self.fc2 = nn.Linear(latent_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc_mean(h), self.fc_logvar(h)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        h = F.relu(self.fc2(z))
        return torch.sigmoid(self.fc3(h))

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        return self.decode(z), mean, logvar


def vae_loss(recon_x, x, mean, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return recon_loss + kld_loss


bottleneck = 3

# Train the VAE
X_train, X_val = train_test_split(data_tensor, test_size=0.2, random_state=42)
train_loader = DataLoader(TensorDataset(X_train, X_train), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, X_val), batch_size=32)

# Initialize the VAE and optimizer
vae_model = VAE(input_dim=10, hidden_dim=6, latent_dim=bottleneck)
optimizer = torch.optim.Adam(vae_model.parameters(), lr=0.0001)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    vae_model.train()
    train_loss = 0
    for batch_data, _ in train_loader:
        optimizer.zero_grad()
        recon_batch, mean, logvar = vae_model(batch_data)
        loss = vae_loss(recon_batch, batch_data, mean, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss / len(X_train):.6f}")

# Extract latent features
vae_model.eval()
with torch.no_grad():
    mean, _ = vae_model.encode(data_tensor)
latent_features = mean.numpy()

# Convert to DataFrame and add goal_status
latent_features_df = pd.DataFrame(latent_features, columns=[f"latent_feature_{i + 1}" for i in range(bottleneck)])
latent_features_df['goal_status'] = data['goal_status'].values

# Save to CSV
latent_features_df.to_csv("/media/nkatz/storage/EVENFLOW-DATA/DFKI/new-3-8-2023/latent_features_vae_dataset.csv",
                          index=False)
