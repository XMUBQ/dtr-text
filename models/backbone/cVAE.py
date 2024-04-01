import torch
import torch.nn as nn


class cVAE(nn.Module):
    def __init__(self, input_dim, condition_dim, n_components, output_dim=None):
        super(cVAE, self).__init__()
        self.encoder_mean = nn.Linear(input_dim + condition_dim, n_components)
        self.encoder_var = nn.Linear(input_dim + condition_dim, n_components)
        self.decoder = nn.Linear(n_components + condition_dim, input_dim if output_dim is None else output_dim)

    @staticmethod
    def reparameterize(z_mean, z_logvar):
        epsilon = torch.randn_like(z_logvar)
        return epsilon * ((0.5 * z_logvar).exp()) + z_mean

    def encode(self, x):
        encoded_mean = self.encoder_mean(x)
        encoded_var = self.encoder_var(x)
        return encoded_mean, encoded_var

    def forward(self, x, condition):
        x = torch.cat((x, condition), dim=1)
        encoded_mean, encoded_var = self.encode(x)
        encoded = self.reparameterize(encoded_mean, encoded_var)

        encoded_label = torch.cat((encoded, condition), dim=1)
        decoded = self.decoder(encoded_label)
        return encoded_mean, encoded_var, decoded
