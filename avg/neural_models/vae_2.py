import torch.nn as nn
import torch


class EnhancedVAE(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, binary=True):
        super(EnhancedVAE, self).__init__()

        self.binary = binary

        # Encoder layers
        encoder_layers = []
        last_dim = input_dim
        for next_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(last_dim, next_dim),
                nn.BatchNorm1d(next_dim),  # Batch normalization
                nn.LeakyReLU(0.2),  # LeakyReLU activation
                nn.Dropout(0.5)  # Dropout for regularization
            ])
            last_dim = next_dim

        self.encoder = nn.Sequential(*encoder_layers)

        self.fc_mean = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

        # Decoder layers
        decoder_layers = []
        last_dim = latent_dim
        for next_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(last_dim, next_dim),
                nn.BatchNorm1d(next_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.5)
            ])
            last_dim = next_dim

        decoder_layers.extend([
            nn.Linear(hidden_dims[0], input_dim),
            nn.Sigmoid()  # To bring the values between 0 and 1
        ])

        self.decoder = nn.Sequential(*decoder_layers)

    # ... [rest of the methods remain unchanged]

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

