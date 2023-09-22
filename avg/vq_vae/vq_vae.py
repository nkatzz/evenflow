import torch
import torch.nn as nn
import torch.nn.functional as F


class VQEmbedding(nn.Module):
    """
    Implements the vector quantization layer.
    """

    def __init__(self, num_embeddings, embedding_dim):
        super(VQEmbedding, self).__init__()

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1. / num_embeddings, 1. / num_embeddings)

    def forward(self, z):
        # Calculate distances between z and embeddings
        distances = (torch.sum(z ** 2, dim=1, keepdim=True)
                     + torch.sum(self.embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(z, self.embedding.weight.t()))

        # Retrieve closest embeddings
        _, indices = distances.min(-1)
        z_q = self.embedding(indices).detach()
        z_q = z_q + (z - z_q).detach()  # ensure gradients flow through

        return z_q, indices


class VQVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_embeddings, embedding_dim):
        super(VQVAE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )

        # Vector Quantization
        self.vq = VQEmbedding(num_embeddings, embedding_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        z_q, indices = self.vq(z)
        x_recon = self.decoder(z_q)

        return x_recon, z, z_q, indices


