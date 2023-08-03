import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, input_dim, n_components):
        super(VAE, self).__init__()
        self.encoder_mean = nn.Linear(input_dim, n_components)
        self.encoder_var = nn.Linear(input_dim, n_components)
        self.decoder = nn.Linear(n_components, input_dim)

    @staticmethod
    def reparameterize(z_mean, z_logvar):
        epsilon = torch.randn_like(z_logvar)
        return epsilon * ((0.5 * z_logvar).exp()) + z_mean

    def forward(self, x):
        encoded_mean = self.encoder_mean(x)
        encoded_var = self.encoder_var(x)
        encoded = self.reparameterize(encoded_mean, encoded_var)
        decoded = self.decoder(encoded)
        return encoded_mean, encoded_var, decoded
