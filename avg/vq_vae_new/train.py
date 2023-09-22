import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from avg.vq_vae_new.vq_vae import *

# Load the dataset
df = pd.read_csv("/media/nkatz/storage/EVENFLOW-DATA/DFKI/new-3-8-2023/DemoDataset_1Robot_clean.csv")

# Splitting data and labels
data = df.drop(columns=['goal_status']).values
labels = df['goal_status'].values

X_train = df.drop('goal_status', axis=1).values
y_train = df['goal_status'].values

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)

# Hyperparameters
input_dim = df.shape[1] - 1  # Exclude the 'goal_status' column
hidden_dim = 256
embedding_dim = 64
batch_size = 64
num_embeddings = 512
num_latent_dims = 8
lr = 0.001
epochs = 100

# Setup data loaders, optimizer, and loss function
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model_multi_dim = VQVAE(input_dim, hidden_dim, num_embeddings, embedding_dim, num_latent_dims)
optimizer = optim.Adam(model_multi_dim.parameters(), lr=0.001)

# Training loop
for epoch in range(epochs):
    for batch in train_loader:
        x = batch[0]
        x_recon, z, z_q, indices = model_multi_dim(x)

        # Compute reconstruction loss
        recon_loss = F.mse_loss(x_recon, x)

        # Compute quantization loss for the multi-dimensional latent representation
        z_q_reshaped = z_q.view(-1, num_latent_dims, embedding_dim)
        quantization_loss = F.mse_loss(z_q_reshaped, z.detach()).sum()

        loss = recon_loss + quantization_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

# Generating discrete latent representation and saving to CSV
model_multi_dim.eval()
all_indices = []
with torch.no_grad():
    for batch in train_loader:
        x = batch[0]
        _, _, _, indices = model_multi_dim(x)
        all_indices.extend(indices.tolist())

df_latent_multi = pd.DataFrame(all_indices, columns=[f"lf_{i + 1}" for i in range(num_latent_dims)])
df_latent_multi['goal_status'] = y_train[:len(all_indices)]
df_latent_multi.to_csv("/media/nkatz/storage/EVENFLOW-DATA/DFKI/new-3-8-2023/latent_features_dataset.csv", index=False)