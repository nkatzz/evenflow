import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import wittgenstein as lw

"""
Similar to vae_ripper, but uses binary latent features in an effort to learn simpler RIPPER models.
This uses a more complex VAE architecture, since the one from vae_simple.py caused the latent space
to collapse to single binary representation for all instances in the dataset (i.e. for a latent 
dimension of 4, every instance in the original dataset of dimensionality 10 was mapped to (1, 0, 0, 0)). 

This uses a staged joint training approach, where the VAE is trained for a few epochs then a RIPPER
model is learnt, the loss is combined with that from the VAE and so on.

In this setting with bottleneck_size = 4 RIPPER takes a ling time to train. Need to try alternatives like
increasing the dimensionality, or trying even more sophisticated architectures.

The RIPPER models have significantly lower predictive performance as compared to the versions learnt from
continuous latent features (i.e. ~ 0.88% avg. accuracy, as opposed to 0.99%...)

Also, need to see what happens if we use the reconstruction of each instance through the entire VAE pipeline,
not just the encoder. We'll have the same number of features, but maybe we'll still be able to learn simpler rules 
"""

data = pd.read_csv('/media/nkatz/storage/EVENFLOW-DATA/DFKI/new-3-8-2023/DemoDataset_1Robot.csv')
features = ['px', 'py', 'pz', 'ox', 'oy', 'oz', 'ow', 'vx', 'vy', 'wz']
X_train = data[features]
y_train = data['goal_status']

# Convert the data to PyTorch tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)


class BinaryVAE(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, binary=True):
        super(BinaryVAE, self).__init__()

        self.binary = binary

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU()
        )
        self.fc_mean = nn.Linear(hidden_dims[1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[1], latent_dim)

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], input_dim),
            nn.Sigmoid()  # To bring the values between 0 and 1
        )

    def encode(self, x):
        h = self.encoder(x)
        mean, logvar = self.fc_mean(h), self.fc_logvar(h)
        if self.binary:
            z = self.reparameterize(mean, logvar)
            binary_z = (z > 0.5).float()  # This step ensures z values are binary
            return binary_z, mean, logvar
        else:
            return mean, logvar

    def reparameterize(self, mean, logvar):
        if self.binary:
            # For binary reparametrization
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mean + eps * std
            logits = torch.sigmoid(z)
            y_soft = logits
            y_hard = (y_soft > 0.5).float()
            y = y_hard - y_soft.detach() + y_soft
            return y
        else:
            # For continuous reparametrization 
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, hard=False):
        if self.binary:
            binary_z, mean, logvar = self.encode(x)
            reconstructed = self.decode(binary_z)
            return reconstructed, mean, logvar
        else:
            mean, logvar = self.encode(x)
            z = self.reparameterize(mean, logvar)
            reconstructed = self.decode(z)
            return reconstructed, mean, logvar


def train_vae_pretraining(vae, data, optimizer, epochs=5):
    """
    Attempt to deal with the latent space collapse by training the VAE for a few epochs
    without binarizing the latent space and then trying to binarize (a standard practice
    that sometimes work in situations like these).

    Not really needed using the more sophisticated architecture of BinaryVAE.
    """
    for epoch in range(epochs):
        optimizer.zero_grad()
        reconstructed, mean, logvar = vae(data)

        # VAE Loss
        reconstruction_loss = F.mse_loss(reconstructed, data, reduction='sum')
        kl_divergence = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        vae_loss = reconstruction_loss + kl_divergence

        vae_loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, VAE Loss: {vae_loss.item()}")


# Training RIPPER in an one-vs-rest fashion
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
            rulesets.append(ripper_clf.ruleset_)
            y_pred = ripper_clf.predict(binary_data)
            accuracy = (y_pred == binary_data['goal_status']).mean()
            accuracies.append(accuracy)
            model_complexity = sum([len(rule) for rule in ripper_clf.ruleset_])
            complexities.append(model_complexity)

    avg_accuracy = np.mean(accuracies) if accuracies else 0
    total_complexity = sum(complexities) if complexities else 0
    print(f'RIPPER Avg accuracy: {avg_accuracy}, Model size: {total_complexity}')
    print(rulesets)
    return avg_accuracy, total_complexity


def train_vae_combined(vae, X_train_tensor, y_train, optimizer, unique_classes, epochs=5, alpha=0.1):
    """
    Combines the VAE loss with that of RIPPER's
    """
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
            latent_df = pd.DataFrame(latent_representation,
                                     columns=[f"latent_{i}" for i in range(latent_representation.shape[1])])
            latent_df['goal_status'] = y_train.values.astype(str)  # Convert goal_status to string

        print('Training RIPPER...')

        avg_accuracy, total_complexity = train_ripper_ovr(latent_df, unique_classes)
        ripper_loss = (1 - avg_accuracy) + total_complexity * alpha
        combined_loss = vae_loss + ripper_loss

        combined_loss.backward()
        optimizer.step()

        print(
            f"Epoch {epoch}/{epochs}, VAE Loss: {vae_loss.item()}, RIPPER Loss: {ripper_loss}, Combined Loss: {combined_loss.item()}")

    # Save the final latent representation of the dataset to CSV
    with torch.no_grad():
        final_latent_representation = vae.encode(X_train_tensor)[0].numpy()
        final_latent_df = pd.DataFrame(final_latent_representation,
                                       columns=[f"latent_{i}" for i in range(final_latent_representation.shape[1])])
        final_latent_df['goal_status'] = y_train.values.astype(str)  # Convert goal_status to string
        final_latent_df.to_csv(
            "/media/nkatz/storage/EVENFLOW-DATA/DFKI/new-3-8-2023/latent_features_vae_ripper_binary.csv", index=False)


# Train the model
vae_model = BinaryVAE(input_dim=10, hidden_dims=[20, 15], latent_dim=4, binary=True)
optimizer = optim.Adam(vae_model.parameters(), lr=0.001)
train_vae_pretraining(vae_model, X_train_tensor, optimizer, epochs=50)
unique_classes = y_train.unique()
train_vae_combined(vae_model, X_train_tensor, y_train, optimizer, unique_classes, epochs=3)
