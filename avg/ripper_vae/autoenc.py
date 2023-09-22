import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from avg.neural.nns import Autoencoder, BinaryAutoencoder, BinaryConcreteAutoencoder, BinaryAutoencoderComplex
from avg.utils.utils import plot_original_vs_reconstructed

"""
Trains a simple autoencoder on the DFKI dataset.
"""

# Load the dataset
data = pd.read_csv('/media/nkatz/storage/EVENFLOW-DATA/DFKI/new-3-8-2023/DemoDataset_1Robot_clean.csv')
features = ['px', 'py', 'pz', 'ox', 'oy', 'oz', 'ow', 'vx', 'vy', 'wz']
data_relevant = data[features]

# Splitting data into training and validation sets
train_data, val_data = train_test_split(data_relevant, test_size=0.2, random_state=42)

# Standardizing the data
# scaler = StandardScaler()
scaler = MinMaxScaler()  # Might help when aiming for binary data
train_data_scaled = scaler.fit_transform(train_data)
val_data_scaled = scaler.transform(val_data)

train_tensors = torch.tensor(train_data_scaled, dtype=torch.float32)
val_tensors = torch.tensor(val_data_scaled, dtype=torch.float32)

train_dataset = TensorDataset(train_tensors, train_tensors)
val_dataset = TensorDataset(val_tensors, val_tensors)

batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

model = Autoencoder(10, 7, 4)
# model = BinaryAutoencoder(10, 10, 9, binary=True)
# model = BinaryAutoencoderComplex(10, 128, 7, binary=True)  # Doesn't work, the loss does not improve
# model = BinaryConcreteAutoencoder(input_dim=10, hidden_dim=10, latent_dim=7, tau=0.5)

# Loss and optimizer
criterion = nn.MSELoss(reduction='mean')
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for batch_features, _ in train_loader:
        optimizer.zero_grad()
        if isinstance(model, Autoencoder):
            outputs = model(batch_features)
        else:  # BinaryAutoencoder
            outputs, _ = model(batch_features)
        loss = criterion(outputs, batch_features)

        # print(batch_features, outputs)

        loss.backward()
        optimizer.step()
        epoch_loss += loss

    divide_by = len(train_dataset) / batch_size
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss.item() / divide_by:.4f}")

    # Visualization of original vs. reconstructed samples after each epoch
    # if (epoch + 1) % 10 == 0:  # Every 10 epochs
    #    plot_original_vs_reconstructed(model, train_tensors, 5)

print('Training completed!')

# After training, get the MSE between each original and reconstructed to 
# see how well the model is trained, just as a sanity check. The two MSEs
# being fairly close is a good indication that our model does not overfit.

# Convert dataset to torch tensor
train_data_tensor = torch.tensor(train_data_scaled, dtype=torch.float32)
val_data_tensor = torch.tensor(val_data_scaled, dtype=torch.float32)

# Get the reconstructed data from the autoencoder
with torch.no_grad():
    if isinstance(model, Autoencoder):
        reconstructed_data_train = model(train_data_tensor)
        reconstructed_data_val = model(val_data_tensor)
    else:
        reconstructed_data_train, _ = model(train_data_tensor)
        reconstructed_data_val, _ = model(val_data_tensor)

# Calculate the MSE
mse_loss_train = F.mse_loss(reconstructed_data_train, train_data_tensor)
mse_loss_val = F.mse_loss(reconstructed_data_val, val_data_tensor)

print(f'Mean Squared Error on the training data: {mse_loss_train.item()}')
print(f'Mean Squared Error on the validation data: {mse_loss_val.item()}')

# Pass the dataset through the autoencoder and save the result to a csv, 
# in order to use it for training other classifiers (e.g. RIPPER)

data_relevant_scaled = scaler.fit_transform(data[features])
data_tensor = torch.tensor(data_relevant_scaled, dtype=torch.float32)


# Extract the encoder from the trained autoencoder


def to_binary(tensor, threshold=0.5):
    return (tensor > threshold).astype(float)


with torch.no_grad():
    if isinstance(model, Autoencoder):
        encoder = nn.Sequential(*list(model.children())[:-1])
        latent_features = encoder(data_tensor).numpy()
    else:
        encoder = nn.Sequential(*list(model.children())[:-2])
        latent_features = model.encode(data_tensor).numpy()  # that's a BinaryAutoencoder, we control if features are binary or not via an arg
        if isinstance(model, BinaryConcreteAutoencoder):
            latent_features = to_binary(latent_features)  # Get hard binary values

# Convert latent features to a DataFrame
latent_features_df = pd.DataFrame(latent_features,
                                  columns=[f"lf_{i + 1}" for i in range(latent_features.shape[1])])

# Add the "goal_status" column to the DataFrame
latent_features_df['goal_status'] = data['goal_status'].values

# Save the DataFrame to a CSV file
csv_path = "/media/nkatz/storage/EVENFLOW-DATA/DFKI/new-3-8-2023/latent_features_dataset.csv"
latent_features_df.to_csv(csv_path, index=False)
print(f"The latent features dataset was saved at {csv_path}")
