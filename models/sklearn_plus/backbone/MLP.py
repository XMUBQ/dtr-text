import torch.nn as nn
import torch

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden=128, num_of_layer=2, step=2):
        super(MLP, self).__init__()

        self.input_size=input_size
        self.output_size=output_size
        self.hidden=hidden
        self.num_of_layer=num_of_layer
        self.step = step

        self.linear = []
        self.linear.append(nn.Linear(in_features=input_size, out_features=hidden))
        for i in range(num_of_layer - 1):
            self.linear.append(nn.BatchNorm1d(hidden))
            self.linear.append(nn.LeakyReLU())
            self.linear.append(nn.Linear(hidden, hidden))
        self.linear.append(nn.BatchNorm1d(hidden))
        self.linear.append(nn.LeakyReLU())
        self.linear.append(nn.Linear(hidden, output_size))
        self.seq_lin = nn.Sequential(*self.linear)


    def forward(self, x):
        return self.seq_lin(x)


