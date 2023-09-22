import torch.nn as nn
import torch.nn.functional as F
import torch


class BinaryConcreteAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, tau=1.0):
        super(BinaryConcreteAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

        # Temperature for Gumbel-Softmax
        self.tau = tau

    def gumbel_softmax(self, logits, eps=1e-10):
        """Sample from the Gumbel-Softmax distribution and optionally discretize."""
        U = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(U + eps) + eps)
        y = logits + gumbel_noise
        return F.softmax(y / self.tau, dim=-1)

    def encode(self, x):
        logits = self.encoder(x)
        return self.gumbel_softmax(logits)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class BinaryAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, binary=True):
        super(BinaryAutoencoder, self).__init__()

        # Encoder layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)

        # Decoder layers
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

        self.binary = binary

    def encode(self, x):
        h = F.relu(self.fc1(x))
        z = torch.sigmoid(self.fc2(h))

        # If binary flag is set, then discretize the latent representation
        if self.binary:
            z = (z > 0.5).float()  # Binary thresholding

        return z

    def decode(self, z):
        h = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


class BinaryAutoencoderComplex(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, binary=True):
        super().__init__()

        # Encoder layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc1a = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc1b = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.fc2 = nn.Linear(hidden_dim // 4, latent_dim)

        # Decoder layers
        self.fc3 = nn.Linear(latent_dim, hidden_dim // 4)
        self.fc3a = nn.Linear(hidden_dim // 4, hidden_dim // 2)
        self.fc3b = nn.Linear(hidden_dim // 2, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

        # Dropout layer
        self.dropout = nn.Dropout(0.5)

        self.binary = binary

    def encode(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc1a(h))
        h = self.dropout(h)
        h = F.relu(self.fc1b(h))
        z = torch.sigmoid(self.fc2(h))

        # If binary flag is set, then discretize the latent representation
        if self.binary:
            z = (z > 0.5).float()  # Binary thresholding

        return z

    def decode(self, z):
        h = F.relu(self.fc3(z))
        h = F.relu(self.fc3a(h))
        h = self.dropout(h)
        h = F.relu(self.fc3b(h))
        return torch.sigmoid(self.fc4(h))

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


class SimpleVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()

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
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        h = F.relu(self.fc2(z))
        return torch.sigmoid(self.fc3(h))

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        return self.decode(z), mean, logvar


class VerySimpleVAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VerySimpleVAE, self).__init__()

        # Encoder layers
        self.fc_mean = nn.Linear(input_dim, latent_dim)
        self.fc_logvar = nn.Linear(input_dim, latent_dim)

        # Decoder layers
        self.fc2 = nn.Linear(latent_dim, input_dim)

    def encode(self, x):
        return self.fc_mean(x), self.fc_logvar(x)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        return torch.sigmoid(self.fc2(z))

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        return self.decode(z), mean, logvar


class BinaryVAE(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, binary=True):
        super().__init__()

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
        z = self.reparameterize(mean, logvar)
        if self.binary:
            binary_z = (z > 0.5).float()  # This step ensures z values are binary
            return binary_z, mean, logvar
        else:
            return z, mean, logvar

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
            binary_z = self.reparameterize(mean, logvar)  # This is redundant (not that we just reparametrized in the encode), but for some reason it improves the performance
            reconstructed = self.decode(binary_z)
            return reconstructed, mean, logvar
        else:
            _, mean, logvar = self.encode(x)
            z = self.reparameterize(mean, logvar)
            reconstructed = self.decode(z)
            return reconstructed, mean, logvar


class EnhancedVAE(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, binary=True):
        super().__init__()

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


class ComplexVAE(nn.Module):
    """
    # Example usage:
      vae_model = ComplexVAE(input_dim=10, latent_dim=3)
      optimizer = torch.optim.Adam(vae_model.parameters(), lr=0.001, weight_decay=1e-5)  # L2 regularization
    """

    def __init__(self, input_dim, latent_dim, dropout_prob=0.5):
        super().__init__()

        # Encoder layers
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)

        # Latent space
        self.fc_mean = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        # Decoder layers
        self.fc4 = nn.Linear(latent_dim, 128)
        self.fc4_dim_adjust = nn.Linear(128, 256)
        self.fc5 = nn.Linear(128, 256)
        self.fc6 = nn.Linear(256, 512)
        self.fc7 = nn.Linear(512, input_dim)

        # Activation function
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout_prob)

        # Additional linear layer to adjust the dimensions
        self.fc1_dim_adjust = nn.Linear(512, 256)

    def encode(self, x):
        h1 = self.dropout(self.leakyrelu(self.fc1(x)))
        h2 = self.dropout(self.leakyrelu(self.fc2(h1)))
        h1_adjusted = self.fc1_dim_adjust(h1)  # Adjust dimension
        h3 = self.dropout(self.leakyrelu(self.fc3(h2 + h1_adjusted)))  # Skip connection
        return self.fc_mean(h3), self.fc_logvar(h3)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        h4 = self.dropout(self.leakyrelu(self.fc4(z)))
        h4_adjusted = self.fc4_dim_adjust(h4)

        h5 = self.dropout(self.leakyrelu(self.fc5(h4)))
        h6 = self.dropout(self.leakyrelu(self.fc6(h5 + h4_adjusted)))  # Skip connection
        return torch.sigmoid(self.fc7(h6))

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        return self.decode(z), mean, logvar
