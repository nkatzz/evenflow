import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from avg.neural_models.autoencoder import Autoencoder

"""
Trains a simple autoencoder on the DFKI dataset.
"""

# Load the dataset
data = pd.read_csv('/media/nkatz/storage/EVENFLOW-DATA/DFKI/new-3-8-2023/DemoDataset_1Robot.csv')
features = ['px', 'py', 'pz', 'ox', 'oy', 'oz', 'ow', 'vx', 'vy', 'wz']
data_relevant = data[features]

# Splitting data into training and validation sets
train_data, val_data = train_test_split(data_relevant, test_size=0.2, random_state=42)

# Standardizing the data
scaler = StandardScaler()
train_data_scaled = scaler.fit_transform(train_data)
val_data_scaled = scaler.transform(val_data)

# Convert data to PyTorch tensors
train_tensors = torch.tensor(train_data_scaled, dtype=torch.float32)
val_tensors = torch.tensor(val_data_scaled, dtype=torch.float32)

train_dataset = TensorDataset(train_tensors, train_tensors)
val_dataset = TensorDataset(val_tensors, val_tensors)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)


model = Autoencoder()

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop
num_epochs = 1000

for epoch in range(num_epochs):
    model.train()
    for batch_features, _ in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_features)
        loss = criterion(outputs, batch_features)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

print('Training completed!')

# After training, get the MSE between each original and reconstructed to 
# see how well the model is trained, just as a sanity check. The two MSEs
# being fairly close is a good indication that our model does not overfit.

# Convert dataset to torch tensor
train_data_tensor = torch.tensor(train_data_scaled, dtype=torch.float32)
val_data_tensor = torch.tensor(val_data_scaled, dtype=torch.float32)

# Get the reconstructed data from the autoencoder
with torch.no_grad():
    reconstructed_data_train = model(train_data_tensor)
    reconstructed_data_val = model(val_data_tensor)

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
encoder = nn.Sequential(*list(model.children())[:-1])

with torch.no_grad():
    latent_features = encoder(data_tensor).numpy()

# Convert latent features to a DataFrame
latent_features_df = pd.DataFrame(latent_features,
                                  columns=[f"latent_feature_{i + 1}" for i in range(latent_features.shape[1])])

# Add the "goal_status" column to the DataFrame
latent_features_df['goal_status'] = data['goal_status'].values

# Save the DataFrame to a CSV file
csv_path = "/media/nkatz/storage/EVENFLOW-DATA/DFKI/new-3-8-2023/latent_features_dataset.csv"
latent_features_df.to_csv(csv_path, index=False)
print(f"The latent features dataset was saved at {csv_path}")
