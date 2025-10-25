#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/25 00:35
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   trainer.py
# @Desc     :   

from torch import nn, no_grad, save, device, abs as torch_abs
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.PT import get_device, TorchDataLoader


class TorchTrainer:
    """ Trainer class for managing training process """

    def __init__(self, model: nn.Module, optimiser, criterion, accelerator: str = "auto"):
        super().__init__()
        """ Initialise the Trainer class
        :param model: the neural network model to be trained
        :param optimiser: the optimiser for updating model parameters
        :param criterion: the loss function
        :param accelerator: device to use for training ("cpu", "cuda", or "auto)
        """
        self._model = model
        self._optimiser = optimiser
        self._criterion = criterion
        self._accelerator = get_device(accelerator)

        # self._train_losses: list[float] = []
        # self._valid_losses: list[float] = []
        # self._accuracies: list[float] = []

    def _epoch_train(self, dataloader: DataLoader | TorchDataLoader) -> float:
        """ Train the model for one epoch
        :param dataloader: DataLoader for training data
        :return: average training loss for the epoch
        """
        # Set model to training mode
        self._model.train()

        _loss: float = 0.0
        _total: float = 0.0
        for i, (features, labels) in enumerate(dataloader):
            features, labels = features.to(device(self._accelerator)), labels.to(device(self._accelerator))

            self._optimiser.zero_grad()
            outputs = self._model(features)
            # print(outputs.shape, labels.shape)
            loss = self._criterion(outputs, labels)
            loss.backward()
            self._optimiser.step()

            _loss += loss.item() * features.size(0)
            _total += features.size(0)

            # print(f"Batches [{i + 1}/{len(dataloader)}] - Train Loss: {_loss / _total:.4f}")

        return _loss / _total

    def _epoch_valid(self, dataloader: DataLoader | TorchDataLoader):
        """ Validate the model for one epoch
        :param dataloader: DataLoader for validation data
        :return: average validation loss for the epoch
        """
        # Set model to evaluation mode
        self._model.eval()

        _loss: float = 0.0
        _total: float = 0.0
        _mae: float = 0.0
        with no_grad():
            for i, (features, labels) in enumerate(dataloader):
                features, labels = features.to(device(self._accelerator)), labels.to(device(self._accelerator))

                outputs = self._model(features)
                # print(outputs.shape, labels.shape)
                loss = self._criterion(outputs, labels)
                _loss += loss.item() * features.size(0)

                # Calculate MAE
                _mae += torch_abs(outputs - labels).sum().item()

                _total += labels.numel()

                # print(
                #     f"Batches [{i + 1}/{len(dataloader)}] - Valid Loss: {_loss / _total:.4f}"
                # )

        return _loss / _total, _mae / _total

    def fit(self,
            train_loader: DataLoader | TorchDataLoader, valid_loader: DataLoader | TorchDataLoader,
            epochs: int, model_save_path: str | None = None
            ) -> None:
        """ Fit the model to the training data
        :param train_loader: DataLoader for training data
        :param valid_loader: DataLoader for validation data
        :param epochs: number of training epochs
        :param model_save_path: path to save the best model parameters
        :return: None
        """
        _best_valid_loss = float("inf")

        for epoch in tqdm(range(epochs), desc="Training Progress", unit="epoches"):
            train_loss = self._epoch_train(train_loader)
            valid_loss, mae = self._epoch_valid(valid_loader)

            # self._train_losses.append(train_loss)
            # self._valid_losses.append(valid_loss)
            # self._accuracies.append(accuracy)

            print(f"Epoch [{epoch + 1}/{epochs}] | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Valid Loss: {valid_loss:.4f} | "
                  f"Valid MAE: {mae:.4f}")

            # Save the model if it has the best validation loss so far
            if valid_loss < _best_valid_loss:
                _best_valid_loss = valid_loss
                save(self._model.state_dict(), model_save_path)
                print(f"Model's parameters saved to {model_save_path}")


if __name__ == "__main__":
    pass
