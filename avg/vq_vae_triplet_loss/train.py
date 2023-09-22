import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from avg.vq_vae_triplet_loss.vq_vae import *
import random
from sklearn.preprocessing import LabelEncoder
import numpy as np


class TripletLoss(nn.Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)
        return F.relu(pos_dist - neg_dist + self.margin).mean()


def sample_positive(y, X, label_to_indices):
    """
    Given a batch of labels, y, sample positive examples from X using pre-computed indices.
    """
    indices = [label_to_indices[label.item()][random.choice(range(len(label_to_indices[label.item()])))] for label in y]
    positive_samples = X[indices]
    return torch.tensor(positive_samples, dtype=torch.float32)


def sample_negative(y, X, label_to_indices):
    """
    Given a batch of labels, y, sample negative examples from X using pre-computed indices.
    """
    all_labels = list(label_to_indices.keys())
    negative_indices = []
    for label in y:
        neg_label = label.item()
        while neg_label == label.item():
            neg_label = random.choice(all_labels)
        negative_indices.append(label_to_indices[neg_label][random.choice(range(len(label_to_indices[neg_label])))])
    negative_samples = X[negative_indices]
    return torch.tensor(negative_samples, dtype=torch.float32)


# Load the dataset
df = pd.read_csv("/media/nkatz/storage/EVENFLOW-DATA/DFKI/new-3-8-2023/DemoDataset_1Robot_clean.csv")

# Hyperparameters
input_dim = df.shape[1] - 1  # Exclude the 'goal_status' column
hidden_dim = 256
embedding_dim = 64
batch_size = 64
num_embeddings = 512
num_latent_dims = 8
lr = 0.001
epochs = 200

X_data = df.drop('goal_status', axis=1).values
y_data = df['goal_status'].values

# Standardize the data
scaler = StandardScaler()
X_data = scaler.fit_transform(X_data)

# Convert class names to integer labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_data)

# Setup data loaders, optimizer, and loss functions
train_dataset = TensorDataset(torch.tensor(X_data, dtype=torch.float32), torch.tensor(y_encoded, dtype=torch.long))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model_multi_dim = VQVAE(input_dim, hidden_dim, num_embeddings, embedding_dim, num_latent_dims)
optimizer = optim.Adam(model_multi_dim.parameters(), lr=0.001)

# Create a dictionary to store indices for each label. This is for speeding-up the sampling
# process by avoiding constantly re-iterating over the entire training set
label_to_indices = {label: np.where(y_encoded == label)[0] for label in np.unique(y_encoded)}

# Training loop with triplet loss integration
for epoch in range(epochs):
    epoch_loss = 0
    for x_anchor, y_anchor in train_loader:
        # Sample positive and negative examples
        x_positive = sample_positive(y_anchor, X_data, label_to_indices)
        x_negative = sample_negative(y_anchor, X_data, label_to_indices)

        # Compute latent representations
        _, z_anchor, _, _ = model_multi_dim(x_anchor)
        _, z_positive, _, _ = model_multi_dim(x_positive)
        _, z_negative, _, _ = model_multi_dim(x_negative)

        # Triplet loss
        triplet_loss = TripletLoss(margin=1.0)
        t_loss = triplet_loss(z_anchor, z_positive, z_negative)

        # VQ-VAE loss components
        x_recon, z, z_q, indices = model_multi_dim(x_anchor)
        recon_loss = F.mse_loss(x_recon, x_anchor)
        z_q_reshaped = z_q.view(-1, num_latent_dims, embedding_dim)
        quantization_loss = F.mse_loss(z_q_reshaped, z.detach()).sum()

        # Combined loss
        total_loss = recon_loss + quantization_loss + t_loss

        # print(recon_loss, quantization_loss, t_loss)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        epoch_loss += total_loss.item()

    print(f"Epoch {epoch + 1}/{epochs}, Epoch Loss: {epoch_loss/len(train_loader)}")

# Generating discrete latent representation and saving to CSV
model_multi_dim.eval()
all_indices = []
with torch.no_grad():
    for batch in train_loader:
        x = batch[0]
        _, _, _, indices = model_multi_dim(x)
        all_indices.extend(indices.tolist())

df_latent_multi = pd.DataFrame(all_indices, columns=[f"lf_{i + 1}" for i in range(num_latent_dims)])
df_latent_multi['goal_status'] = y_data[:len(all_indices)]
df_latent_multi.to_csv("/media/nkatz/storage/EVENFLOW-DATA/DFKI/new-3-8-2023/latent_features_dataset.csv", index=False)
