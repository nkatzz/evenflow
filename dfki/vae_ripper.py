import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import wittgenstein as lw

"""
Trains a VAE on the DFKI dataset, recosntructs the dataset through the encoder, trains RIPPER
on the lower-dimentionality latent features and combines RIPPER's loss with the VAE loss in a 
joint training loop.
"""

# 1. Load the data
data = pd.read_csv('/media/nkatz/storage/EVENFLOW-DATA/DFKI/new-3-8-2023/DemoDataset_1Robot.csv')
features = ['px', 'py', 'pz', 'ox', 'oy', 'oz', 'ow', 'vx', 'vy', 'wz']
X_train = data[features]
y_train = data['goal_status']

# 2. Convert the data to PyTorch tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)

# 3. VAE Definition
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
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mean + eps*std

    def decode(self, z):
        h = F.relu(self.fc2(z))
        return torch.sigmoid(self.fc3(h))

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        return self.decode(z), mean, logvar

# 4. Training RIPPER in an OvA fashion
def train_ripper_ovr(data, unique_classes):
    rulesets = []
    accuracies = []
    complexities = []

    for cls in unique_classes:
        ripper_clf = lw.RIPPER()
        binary_data = data.copy()
        binary_data['goal_status'] = binary_data['goal_status'].apply(lambda x: 1 if x == cls else 0)
        
        if binary_data['goal_status'].sum() > 0:
            ripper_clf.fit(binary_data, class_feat='goal_status', pos_class=1)
            rulesets.append(ripper_clf)
            y_pred = ripper_clf.predict(binary_data)
            accuracy = (y_pred == binary_data['goal_status']).mean()
            accuracies.append(accuracy)
            model_complexity = sum([len(rule) for rule in ripper_clf.ruleset_])
            complexities.append(model_complexity)

    avg_accuracy = np.mean(accuracies) if accuracies else 0
    total_complexity = sum(complexities) if complexities else 0
    return avg_accuracy, total_complexity

# 5. Training Loop for VAE + RIPPER
def train_vae_combined(vae, X_train_tensor, y_train, optimizer, unique_classes, epochs=5, alpha=0.1):
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        reconstructed, mean, logvar = vae(X_train_tensor)
        
        # VAE Loss
        reconstruction_loss = F.mse_loss(reconstructed, X_train_tensor, reduction='sum')
        kl_divergence = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        vae_loss = reconstruction_loss + kl_divergence
        
        # Extracting latent representation for RIPPER
        with torch.no_grad():
            latent_representation = vae.encode(X_train_tensor)[0].numpy()
            latent_df = pd.DataFrame(latent_representation, columns=[f"latent_{i}" for i in range(latent_representation.shape[1])])
            latent_df['goal_status'] = y_train.values.astype(str)  # Convert goal_status to string

        avg_accuracy, total_complexity = train_ripper_ovr(latent_df, unique_classes)
        print(f'Avg ac: {avg_accuracy}, compl: {total_complexity}')
        ripper_loss = (1 - avg_accuracy) + total_complexity * alpha
        combined_loss = vae_loss + ripper_loss
        
        combined_loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}/{epochs}, VAE Loss: {vae_loss.item()}, RIPPER Loss: {ripper_loss}, Combined Loss: {combined_loss.item()}")

    # Save the final latent representation of the dataset to CSV
    with torch.no_grad():
        final_latent_representation = vae.encode(X_train_tensor)[0].numpy()
        final_latent_df = pd.DataFrame(final_latent_representation, columns=[f"latent_{i}" for i in range(final_latent_representation.shape[1])])
        final_latent_df['goal_status'] = y_train.values.astype(str)  # Convert goal_status to string
        final_latent_df.to_csv("/media/nkatz/storage/EVENFLOW-DATA/DFKI/new-3-8-2023/latent_features_vae_ripper.csv", index=False)

vae_model = VAE(input_dim=10, hidden_dim=20, latent_dim=4)
optimizer = optim.Adam(vae_model.parameters(), lr=0.001)
unique_classes = y_train.unique()
train_vae_combined(vae_model, X_train_tensor, y_train, optimizer, unique_classes, epochs=1000)


