import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(10, 7),
            nn.ReLU(),
            nn.Linear(7, 5),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(5, 7),
            nn.ReLU(),
            nn.Linear(7, 10)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
