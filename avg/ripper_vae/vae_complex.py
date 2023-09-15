import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from avg.neural_models.vae_1 import BinaryVAE
from avg.ripper_vae.vae_simple import vae_loss

"""
Similar to vae_simple.py, but uses a more sophisticated VAE architecture
"""

# 1. Load and preprocess data
data = pd.read_csv('/media/nkatz/storage/EVENFLOW-DATA/DFKI/new-3-8-2023/DemoDataset_1Robot.csv')
scaler = StandardScaler()
data_relevant_scaled = scaler.fit_transform(data[['px', 'py', 'pz', 'ox', 'oy', 'oz', 'ow', 'vx', 'vy', 'wz']])
data_tensor = torch.tensor(data_relevant_scaled, dtype=torch.float32)


if __name__ == "__main__":
    bottleneck = 3

    # Train the VAE
    X_train, X_val = train_test_split(data_tensor, test_size=0.2, random_state=42)
    train_loader = DataLoader(TensorDataset(X_train, X_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, X_val), batch_size=32)

    # Initialize the VAE and optimizer
    vae_model = BinaryVAE(input_dim=10, hidden_dims=[20, 15], latent_dim=bottleneck, binary=False)
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

        # Notice that the expected behavior is that the MSE loss tends to increase, while KLD tends to increase
        # over time. This is because in the initial stages of training, the VAE focuses on learning to reconstruct
        # the input data, which leads to a decrease in the reconstruction (MSE) loss. As the network gets better
        # at encoding the nuances of the input data into the latent space and decoding it back, this loss continues
        # to decrease. The KL divergence measures how much the learned latent variable distribution deviates from
        # a prior distribution (typically a standard normal distribution). Initially, the VAE might not focus as
        # much on regularizing the latent space, which might keep the KLD relatively low. However, as the VAE becomes
        # better at reconstruction, it starts to focus more on the regularization term, i.e., ensuring that the latent
        # variable distribution matches the prior. This leads to an increase in the KL divergence, especially if the
        # latent space was capturing complex patterns that don't align with a simple standard normal distribution.
        # This is the trade-off inherent in VAEs: as you push for a better reconstruction, you might be pushing your
        # latent distribution away from the prior, leading to an increase in KLD.
        # The decrease in MSE loss and the increase in KLD indicate the model's attempt to balance the twin objectives
        # of accurate data reconstruction and regularization of the latent space to adhere to a prior distribution.
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss / len(X_train):.6f}, MSE loss: {recon_loss/len(X_train)}, KLD: {kld/len(X_train)}")

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
