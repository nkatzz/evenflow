import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from avg.neural.nns import VerySimpleVAE, BinaryVAE
from avg.utils.utils import plot_original_vs_reconstructed


# Training function
def train_vae(vae, data_loader, optimizer, epochs=20):
    vae.train()
    for epoch in range(epochs):
        epoch_loss, epoch_recon_loss, epoch_kl_loss = 0.0, 0.0, 0.0

        for batch_idx, (data, _) in enumerate(data_loader):
            optimizer.zero_grad()
            reconstructed, mean, logvar = vae(data)

            reconstruction_loss = F.mse_loss(reconstructed, data, reduction='mean')
            kl_divergence = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

            # Don't call .item() on the values, since this will break the computation graph and backprop won't work
            batch_loss = reconstruction_loss + 0.0001 * kl_divergence

            batch_loss.backward()
            optimizer.step()

            # It's OK to detach the tensors from the computational graph here (by calling .item()), since we don't
            # and it also saves computational resources.
            epoch_loss += batch_loss.item() * batch_size
            epoch_recon_loss += reconstruction_loss.item()
            epoch_kl_loss += kl_divergence.item()

        divide_by = len(data_loader.dataset) / batch_size
        print(f"Epoch [{epoch + 1}/{epochs}] MSE Loss: {epoch_recon_loss / divide_by}, "
              f"KLD: {epoch_kl_loss / len(data_loader.dataset)}")

        # Visualization of original vs. reconstructed samples after each epoch

        if visualize_reconstructed and (epoch + 1) % 10 == 0:
            plot_original_vs_reconstructed(vae, data_tensor, 5)


visualize_reconstructed = True

data = pd.read_csv('/media/nkatz/storage/EVENFLOW-DATA/DFKI/new-3-8-2023/DemoDataset_1Robot_clean.csv')
# features = ['px', 'py', 'pz', 'ox', 'oy', 'oz', 'ow', 'vx', 'vy', 'wz']

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# scaler = StandardScaler()
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(X)
data_tensor = torch.tensor(scaled_data, dtype=torch.float32)

batch_size = 32

# Splitting the dataset into training and validation sets
X_train, X_val = train_test_split(data_tensor, test_size=0.2, random_state=42)
train_loader = DataLoader(TensorDataset(X_train, X_train), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, X_val), batch_size=batch_size)

# Create and train the VAE
input_dim = 10  # len(features)
hidden_dims = [8, 8]  # 20, 15
latent_dim = 5

vae = BinaryVAE(input_dim, hidden_dims, latent_dim, binary=True)
# vae = SimpleVAE(input_dim, 7, latent_dim)
# vae = ComplexVAE(input_dim, latent_dim)
# vae = VerySimpleVAE(input_dim, latent_dim)

optimizer = optim.Adam(vae.parameters(), lr=0.0001)

train_vae(vae, train_loader, optimizer, epochs=100)

with torch.no_grad():
    latent_features, _, _ = vae.encode(data_tensor)
    latent_features = latent_features.numpy()

# Convert to Pandas DataFrame
latent_features_df = pd.DataFrame(latent_features, columns=[f"lf_{i + 1}" for i in range(latent_features.shape[1])])
labels_column_name = y.name
latent_features_df[labels_column_name] = data[labels_column_name].values

# Save to CSV
latent_features_df.to_csv("/media/nkatz/storage/EVENFLOW-DATA/DFKI/new-3-8-2023/latent_features_dataset.csv",
                          index=False)
