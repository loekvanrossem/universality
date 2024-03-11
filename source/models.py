from typing import Optional

import torch
from torch import nn
from torch.optim import Optimizer

from preprocessing import Encoding, Direct


class MLP(nn.ModuleList):
    """
    A simple non-linear feedforward DNN

    Attributes
    ----------
    input_size : int
        The input dimension
    output_size : int
        The output dimension
    hidden_dim : int
        The number of hidden neurons per hidden layer
    n_hid_layers : int
        The number of hidden layers
    device : device
        The device to put the model on
    init_std : float, default 1
        The standard deviation of the initial weight distribution
    encoding : Encoding
        An encoding from input symbols to neural activities
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_dim: int,
        n_hid_layers: int,
        device: torch.device,
        init_std: float = 1,
        non_linearity=torch.nn.functional.leaky_relu,
        encoding: Optional[Encoding] = None,
    ):
        super(MLP, self).__init__()

        self.device = device
        self.non_linearity = non_linearity

        if encoding is None:
            self.encoding = Direct()
        else:
            self.encoding = encoding

        # Defining the layers
        for n in range(n_hid_layers + 1):
            dim_in, dim_out = hidden_dim, hidden_dim
            if n == 0:
                dim_in = input_size
            if n == n_hid_layers:
                dim_out = output_size

            self.append(nn.Linear(dim_in, dim_out, bias=True))

        # Initialize the parameters
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.xavier_normal_(mod.weight, gain=init_std)
                # nn.init.normal_(mod.weight, std=init_std)
                nn.init.zeros_(mod.bias)

        self.to(device)

    def forward(self, x):
        a = x
        activations = []
        for n, layer in enumerate(self):
            a = layer(a)
            if n != len(self) - 1:
                if self.non_linearity != None:
                    a = self.non_linearity(a)
            activations.append(a)

        output = activations.pop()

        return output, activations

    def train_step(self, optimizer: Optimizer, criterion, dataloader):
        self.train()
        av_loss = 0
        for batch in dataloader:
            inputs, outputs = batch
            optimizer.zero_grad()

            output, activations = self(inputs)

            loss = criterion(torch.squeeze(output), torch.squeeze(outputs))

            loss.backward()
            optimizer.step()
            av_loss += loss / len(dataloader)
        return av_loss


class CNN(MLP):
    def __init__(
        self,
        encoding: Encoding,
        input_size: int,
        output_size: int,
        hidden_dim: int,
        n_hid_layers: int,
        device: torch.device,
        init_std: float = 1,
        non_linearity=torch.nn.functional.leaky_relu,
    ):
        super(MLP, self).__init__()

        kernel_size = 11
        n_channels = 20

        self.device = device
        self.non_linearity = non_linearity

        self.encoding = encoding

        # Defining the layers
        self.append(nn.Linear(input_size, hidden_dim, bias=True))
        self.append(
            nn.Conv1d(
                1,
                n_channels,
                kernel_size=kernel_size,
                stride=1,
                # padding=int((kernel_size - 1) / 2),
                bias=True,
            )
        )
        for n in range(n_hid_layers - 1):
            self.append(
                nn.Conv1d(
                    n_channels,
                    n_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    # padding=int((kernel_size - 1) / 2),
                    bias=True,
                )
            )
        self.append(
            nn.Conv1d(
                n_channels,
                1,
                kernel_size=kernel_size,
                stride=1,
                # padding=int((kernel_size - 1) / 2),
                bias=True,
            )
        )
        self.append(
            nn.Linear(hidden_dim - kernel_size * n_hid_layers, output_size, bias=True)
        )

        # Initialize the parameters
        for mod in self.modules():
            if isinstance(mod, nn.Conv1d) or isinstance(mod, nn.Linear):
                nn.init.xavier_normal_(mod.weight, gain=init_std)
                # nn.init.kaiming_normal_(mod.weight, nonlinearity="leaky_relu")
                nn.init.zeros_(mod.bias)

        self.to(device)

    def forward(self, x):
        x = x.unsqueeze(1)

        a = x
        activations = []
        for n, layer in enumerate(self):
            a = layer(a)
            if n != len(self) - 1:
                if self.non_linearity is not None:
                    a = self.non_linearity(a)
            activations.append(a)

        activations = [
            layer.squeeze().reshape((x.shape[0], -1)) for layer in activations
        ]

        output = activations.pop()

        return output, activations


class ResNet(MLP):
    def __init__(
        self,
        encoding: Encoding,
        input_size: int,
        output_size: int,
        hidden_dim: int,
        n_hid_layers: int,
        device: torch.device,
        init_std: float = 1,
        non_linearity=torch.nn.functional.leaky_relu,
    ):
        super(MLP, self).__init__()

        self.device = device
        self.non_linearity = non_linearity

        self.encoding = encoding

        # Defining the layers
        for n in range(n_hid_layers + 1):
            dim_in, dim_out = hidden_dim, hidden_dim
            if n == 0:
                dim_in = input_size
            if n == n_hid_layers:
                dim_out = output_size

            self.append(nn.Linear(dim_in, dim_out, bias=True))

        # Initialize the parameters
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.xavier_normal_(mod.weight, gain=init_std)
                nn.init.zeros_(mod.bias)

        self.to(device)

    def forward(self, x):
        a = x
        activations = []
        for n, layer in enumerate(self):
            a = layer(a)
            if n != len(self) - 1:
                if self.non_linearity is not None:
                    a = self.non_linearity(a)
            if 1 < n < len(self) - 1:
                a = a + activations[-2]
            activations.append(a)

        output = activations.pop()

        return output, activations


class Dropout(MLP):
    def __init__(
        self,
        encoding: Encoding,
        input_size: int,
        output_size: int,
        hidden_dim: int,
        n_hid_layers: int,
        device: torch.device,
        init_std: float = 1,
        non_linearity=torch.nn.functional.leaky_relu,
    ):
        super(MLP, self).__init__()

        self.device = device
        self.non_linearity = non_linearity

        self.encoding = encoding

        # Defining the layers
        for n in range(n_hid_layers + 1):
            dim_in, dim_out = hidden_dim, hidden_dim
            if n == 0:
                dim_in = input_size
            if n == n_hid_layers:
                dim_out = output_size

            self.append(nn.Dropout(p=0.1))
            self.append(nn.Linear(dim_in, dim_out, bias=True))

        # Initialize the parameters
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.xavier_normal_(mod.weight, gain=init_std)
                nn.init.zeros_(mod.bias)

        self.to(device)

    def forward(self, x):
        a = x
        activations = []
        for n, layer in enumerate(self):
            a = layer(a)
            if n != len(self) - 1:
                if isinstance(layer, nn.Linear) and self.non_linearity is not None:
                    a = self.non_linearity(a)
            activations.append(a)

        output = activations.pop()

        return output, activations
