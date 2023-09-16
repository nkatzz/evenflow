import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from avg.neural_models.vae_simple import SimpleVAE

"""
Trains a simple VAE on the DFKI dataset.
"""

data = pd.read_csv('/media/nkatz/storage/EVENFLOW-DATA/DFKI/new-3-8-2023/DemoDataset_1Robot.csv')
scaler = StandardScaler()
data_relevant_scaled = scaler.fit_transform(data[['px', 'py', 'pz', 'ox', 'oy', 'oz', 'ow', 'vx', 'vy', 'wz']])
data_tensor = torch.tensor(data_relevant_scaled, dtype=torch.float32)


def vae_loss(recon_x, x, mean, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return recon_loss, kld_loss


if __name__ == "__main__":
    bottleneck = 3

    X_train, X_val = train_test_split(data_tensor, test_size=0.2, random_state=42)
    train_loader = DataLoader(TensorDataset(X_train, X_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, X_val), batch_size=32)

    # Initialize the VAE and optimizer
    vae_model = SimpleVAE(input_dim=10, hidden_dim=6, latent_dim=bottleneck)
    optimizer = torch.optim.Adam(vae_model.parameters(), lr=0.0001)

    # Training loop
    num_epochs = 1000
    for epoch in range(num_epochs):
        vae_model.train()
        train_loss = 0
        recon_loss = 0
        kld = 0
        for batch_data, _ in train_loader:
            optimizer.zero_grad()
            recon_batch, mean, logvar = vae_model(batch_data)
            loss_1, loss_2 = vae_loss(recon_batch, batch_data, mean, logvar)
            loss = loss_1 + loss_2
            train_loss += loss.item()
            recon_loss += loss_1
            kld += loss_2
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss / len(X_train):.6f}, "
              f"MSE loss: {recon_loss/len(X_train)}, KLD: {kld/len(X_train)}")

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
