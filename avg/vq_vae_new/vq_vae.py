import torch
import torch.nn as nn
import torch.nn.functional as F


class VQEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(VQEmbedding, self).__init__()

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1. / num_embeddings, 1. / num_embeddings)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

    def forward(self, z):
        # Calculate distances from embedding vectors
        distances = (z ** 2).sum(dim=-1, keepdim=True) + (self.embedding.weight ** 2).sum(dim=1) \
                    - 2 * z @ self.embedding.weight.t()
        _, indices = distances.min(dim=-1)
        z_q = self.embedding(indices)
        return z_q, indices


class VQVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_embeddings, embedding_dim, num_latent_dims):
        super(VQVAE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_latent_dims * embedding_dim)
        )

        self.num_latent_dims = num_latent_dims
        self.embedding_dim = embedding_dim

        # Vector Quantization
        self.vq = VQEmbedding(num_embeddings, embedding_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(num_latent_dims * embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        z = z.view(-1, self.num_latent_dims, self.embedding_dim)

        # Pass each latent dimension through VQ layer
        quantized_outputs = []
        all_indices = []
        for i in range(self.num_latent_dims):
            z_q, indices = self.vq(z[:, i, :])
            quantized_outputs.append(z_q.unsqueeze(1))
            all_indices.append(indices)

        z_q_combined = torch.cat(quantized_outputs, dim=1).view(-1, self.num_latent_dims * self.embedding_dim)
        indices_combined = torch.stack(all_indices, dim=1)

        x_recon = self.decoder(z_q_combined)

        return x_recon, z, z_q_combined, indices_combined

