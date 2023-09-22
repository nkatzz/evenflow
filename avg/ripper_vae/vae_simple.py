import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from avg.neural.nns import SimpleVAE

"""
Trains a simple VAE on the DFKI dataset.
"""

data = pd.read_csv('/media/nkatz/storage/EVENFLOW-DATA/DFKI/new-3-8-2023/DemoDataset_1Robot.csv')
scaler = StandardScaler()
data_relevant_scaled = scaler.fit_transform(data[['px', 'py', 'pz', 'ox', 'oy', 'oz', 'ow', 'vx', 'vy', 'wz']])
data_tensor = torch.tensor(data_relevant_scaled, dtype=torch.float32)


def vae_loss(recon_x, x, mean, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    kld_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return recon_loss, kld_loss


if __name__ == "__main__":
    bottleneck = 4

    X_train, X_val = train_test_split(data_tensor, test_size=0.2, random_state=42)
    train_loader = DataLoader(TensorDataset(X_train, X_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, X_val), batch_size=32)

    vae_model = SimpleVAE(input_dim=10, hidden_dim=7, latent_dim=bottleneck)
    optimizer = torch.optim.Adam(vae_model.parameters(), lr=0.0001)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)

    # Training loop
    num_epochs = 1000
    for epoch in range(num_epochs):
        vae_model.train()

        optimizer.zero_grad()
        recon, mean, logvar = vae_model(X_train_tensor)
        mse, kld = vae_loss(recon, X_train_tensor, mean, logvar)
        loss = mse + kld / len(X_train_tensor)
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {loss:.6f}, "
              f"MSE loss: {mse}, KLD: {kld / len(X_train_tensor)}")

        # This uses mini-batches. It is much less efficient due to the overhead of handling the
        # mini-batches, without any obvious regularization advantage
        """
        train_loss = []
        recon_loss = []
        kld_loss = []
        for batch, _ in train_loader:
            optimizer.zero_grad()
            recon_batch, mean, logvar = vae_model(batch)
            mse, kld = vae_loss(recon_batch, batch, mean, logvar)
            loss = mse + kld
            train_loss.append(loss.item())
            recon_loss.append(mse)
            kld_loss.append(kld.item())
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {sum(train_loss) / len(train_loss):.6f}, "
              f"MSE loss: {sum(recon_loss)/len(recon_loss)}, KLD: {sum(kld_loss)/len(kld_loss)}")
        """

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
