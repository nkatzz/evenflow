import torch.nn as nn
import torch


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
