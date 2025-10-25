#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/25 00:39
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   models.py
# @Desc     :   

from torch import nn
from torchsummary import summary


class RegressionTorchModel(nn.Module):
    """ A simple fully connected neural network model """

    def __init__(self, features: int, hidden_units: int, out_size: int, dropout_rate: float = 0.2):
        """ Initialise the LinearTorchModel class
        :param features: number of input features
        :param hidden_units: number of neurons in the hidden layer
        :param out_size: number of output neurons (e.g., 1 for regression)
        :param dropout_rate: dropout rate for regularization
        """
        super(RegressionTorchModel, self).__init__()
        self._features = features

        self._model = nn.Sequential(
            nn.Linear(self._features, hidden_units),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_units, hidden_units // 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_units // 2, out_size)
        )

        self._model.apply(self._initialisation)

    @staticmethod
    def _initialisation(layer):
        """ Initialise the weights of a layer
        :param layer: the layer to be initialised
        """
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="leaky_relu")
            nn.init.zeros_(layer.bias)

    def forward(self, X):
        """ Forward pass of the model
        :param X: input tensor
        :return: output tensor
        """
        return self._model(X)

    def summary(self):
        """ Print the model summary """
        summary(self, (self._features,))
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Model Summary for {self.__class__.__name__}")
        print(self)
        print("-" * 64)


if __name__ == "__main__":
    pass
