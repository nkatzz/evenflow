import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from avg.vq_vae.vq_vae import *

# Load the dataset
df = pd.read_csv("/media/nkatz/storage/EVENFLOW-DATA/DFKI/new-3-8-2023/DemoDataset_1Robot_clean.csv")

# Splitting data and labels
data = df.drop(columns=['goal_status']).values
labels = df['goal_status'].values

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# Create data loaders
batch_size = 64
train_dataset = TensorDataset(X_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Define VQ-VAE architecture (as provided above)

# Hyperparameters
input_dim = df.shape[1] - 1  # Exclude the 'goal_status' column
hidden_dim = 256
embedding_dim = 64
num_embeddings = 512
lr = 0.001
epochs = 20

model = VQVAE(input_dim, hidden_dim, num_embeddings, embedding_dim)
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training loop
for epoch in range(epochs):
    for batch in train_loader:
        x = batch[0]
        x_recon, z, z_q, _ = model(x)

        # Compute reconstruction loss
        recon_loss = F.mse_loss(x_recon, x)

        # Compute quantization loss
        quantization_loss = F.mse_loss(z_q, z.detach())  # only gradient for encoder

        loss = recon_loss + quantization_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

# Generating discrete latent representation
model.eval()
with torch.no_grad():
    z_indices = []
    for batch in train_loader:
        x = batch[0]
        x_recon, z, z_q, indices = model(x)
        z_indices.extend(indices.tolist())



# Saving to CSV


df_latent = pd.DataFrame({'latent_representation': z_indices, 'goal_status': y_train})
df_latent.to_csv("latent_representation.csv", index=False)
