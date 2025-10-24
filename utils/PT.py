#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/25 00:36
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   PT.py
# @Desc     :   

from numpy import ndarray, random as np_random
from pandas import DataFrame, Series
from random import seed as rnd_seed, getstate, setstate
from torch import (cuda, backends, Tensor, tensor, float32, int64,
                   manual_seed, get_rng_state, set_rng_state)

from torch.utils.data import Dataset, DataLoader
from typing import Union, Any

from utils.decorator import timer


class TorchRandomSeed:
    """ Setting random seed for reproducibility """

    def __init__(self, description: str, seed: int = 27):
        """ Initialise the RandomSeed class
        :param description: the description of a random seed
        :param seed: the seed value to be set
        """
        self._description: str = description
        self._seed: int = seed
        self._previous_py_seed = None
        self._previous_pt_seed = None
        self._previous_np_seed = None

    def __enter__(self):
        """ Set the random seed """
        # Save the previous random seed state
        self._previous_py_seed = getstate()
        self._previous_pt_seed = get_rng_state()
        self._previous_np_seed = np_random.get_state()

        # Set the new random seed
        rnd_seed(self._seed)
        manual_seed(self._seed)
        np_random.seed(self._seed)

        print("*" * 50)
        print(f"{self._description!r} has been set randomness {self._seed}.")
        print("-" * 50)

        return self

    def __exit__(self, *args):
        """ Exit the random seed context manager """
        # Restore the previous random seed state
        if self._previous_py_seed is not None:
            setstate(self._previous_py_seed)
        if self._previous_pt_seed is not None:
            set_rng_state(self._previous_pt_seed)
        if self._previous_np_seed is not None:
            np_random.set_state(self._previous_np_seed)

        print("-" * 50)
        print(f"{self._description!r} has been restored to previous randomness.")
        print("*" * 50)
        print()

    def __repr__(self):
        """ Return a string representation of the random seed """
        return f"{self._description!r} is set to randomness {self._seed}."


@timer
def check_device() -> None:
    """ Check Available Device (CPU, GPU, MPS)
    :return: dictionary of available devices
    """

    # CUDA (NVIDIA GPU)
    if cuda.is_available():
        count: int = cuda.device_count()
        print(f"Detected {count} CUDA GPU(s):")
        for i in range(count):
            print(f"GPU {i}: {cuda.get_device_name(i)}")
            print(f"- Memory Usage:")
            print(f"- Allocated: {round(cuda.memory_allocated(i) / 1024 ** 3, 1)} GB")
            print(f"- Cached:    {round(cuda.memory_reserved(i) / 1024 ** 3, 1)} GB")

    # MPS (Apple Silicon GPU)
    elif backends.mps.is_available():
        print("Apple MPS device detected.")

    # Fallback: CPU
    else:
        print("Due to GPU or MPS unavailable, using CPU.")


@timer
def get_device(accelerator: str = "auto", cuda_mode: int = 0) -> str:
    """ Get the appropriate device based on the target device string
    :param accelerator: the target device string ("auto", "cuda", "mps", "cpu")
    :param cuda_mode: the CUDA device index to use (if applicable)
    :return: the appropriate device string
    """
    match accelerator:
        case "auto":
            if cuda.is_available():
                count: int = cuda.device_count()
                print(f"Detected {count} CUDA GPU(s):")
                if cuda_mode < count:
                    for i in range(count):
                        print(f"GPU {i}: {cuda.get_device_name(i)}")
                        print(f"- Memory Usage:")
                        print(f"- Allocated: {round(cuda.memory_allocated(i) / 1024 ** 3, 1)} GB")
                        print(f"- Cached:    {round(cuda.memory_reserved(i) / 1024 ** 3, 1)} GB")
                    print(f"The current accelerator is set to cuda:{cuda_mode}.")
                    return f"cuda:{cuda_mode}"
                else:
                    print(f"CUDA device index {cuda_mode} is out of range. Using 'cuda:0' instead.")
                    return "cuda:0"
            elif backends.mps.is_available():
                print("Apple MPS device detected.")
                return "mps"
            else:
                print("Due to GPU or MPS unavailable, using CPU ).")
                return "cpu"
        case "cuda":
            if cuda.is_available():
                count: int = cuda.device_count()
                print(f"Detected {count} CUDA GPU(s):")
                if cuda_mode < count:
                    for i in range(count):
                        print(f"GPU {i}: {cuda.get_device_name(i)}")
                        print(f"- Memory Usage:")
                        print(f"- Allocated: {round(cuda.memory_allocated(i) / 1024 ** 3, 1)} GB")
                        print(f"- Cached:    {round(cuda.memory_reserved(i) / 1024 ** 3, 1)} GB")
                    print(f"The current accelerator is set to cuda:{cuda_mode}.")
                    return f"cuda:{cuda_mode}"
                else:
                    print(f"CUDA device index {cuda_mode} is out of range. Using 'cuda:0' instead.")
                    return "cuda:0"
            else:
                print("Due to GPU unavailable, using CPU.")
                return "cpu"
        case "mps":
            if backends.mps.is_available():
                print("Apple MPS device detected.")
                return "mps"
            else:
                print("Due to MPS unavailable, using CPU.")
                return "cpu"
        case "cpu":
            print("Using CPU as target device.")
            return "cpu"

        case _:
            print("Due to GPU unavailable, using CPU.")
            return "cpu"


@timer
def arr2tensor(data: ndarray, accelerator: str, is_grad: bool = False) -> Tensor:
    """ Convert a NumPy array to a PyTorch tensor
    :param data: the NumPy array to be converted
    :param accelerator: the device to place the tensor on
    :param is_grad: whether the tensor requires gradient computation
    :return: the converted PyTorch tensor
    """
    return tensor(data, dtype=float32, device=accelerator, requires_grad=is_grad)


@timer
def df2tensor(data: DataFrame, is_label: bool = False, accelerator: str = "cpu", is_grad: bool = False) -> Tensor:
    """ Convert a Pandas DataFrame to a PyTorch tensor
    :param data: the DataFrame to be converted
    :param is_label: whether the DataFrame requires label
    :param accelerator: the device to place the tensor on
    :param is_grad: whether the tensor requires gradient computation
    :return: the converted PyTorch tensor
    """
    if is_label:
        t: Tensor = tensor(data.values, dtype=int64, device=accelerator, requires_grad=is_grad)
    else:
        t: Tensor = tensor(data.values, dtype=float32, device=accelerator, requires_grad=is_grad)

    print(f"The tensor shape is {t.shape}, and its dtype is {t.dtype}.")

    return t


class GrayTensorReshaper:

    def __init__(self, flat: Tensor):
        self._height: int = 28
        self._width: int = 28
        self._channels: int = 1
        self._image: Tensor = flat.reshape(-1, self._channels, self._height, self._width)

    @property
    def height(self) -> int:
        return self._height

    @property
    def width(self) -> int:
        return self._width

    @property
    def channels(self) -> int:
        return self._channels

    @property
    def shape(self) -> tuple:
        return self._image.shape

    def __call__(self) -> Tensor:
        return self._image

    def __getitem__(self, index: int) -> Tensor:
        return self._image[index]

    def __len__(self) -> int:
        return len(self._image)

    def __repr__(self) -> str:
        return f"GrayTensorReshape({self._image.shape})"


class LabelTorchDataset(Dataset):
    """ A custom PyTorch Dataset class for handling label features and labels """

    def __init__(self, features: Any, labels: Any):
        """ Initialise the TorchDataset class
        :param features: the feature tensor
        :param labels: the label tensor
        """
        self._features: Tensor = self._to_tensor(features)
        self._labels: Tensor = self._to_tensor(labels, is_label=True)

    @property
    def features(self) -> Tensor:
        """ Return the feature tensor as a property """
        return self._features

    @property
    def labels(self) -> Tensor:
        """ Return the label tensor as a property """
        return self._labels

    @staticmethod
    def _to_tensor(data: Union[DataFrame, Tensor, ndarray, list], is_label: bool = False) -> Tensor:
        """ Convert input data to a PyTorch tensor on the specified device
        :param data: the input data (DataFrame, ndarray, list, or Tensor)
        :param is_label: whether the data is label data
        :return: the converted PyTorch tensor
        """
        if isinstance(data, (DataFrame, Series)):
            out = tensor(data.values, dtype=int64 if is_label else float32)
        elif isinstance(data, Tensor):
            out = data.long() if is_label else data.float()
        elif isinstance(data, (ndarray, list)):
            out = tensor(data, dtype=int64 if is_label else float32)
        elif isinstance(data, GrayTensorReshaper):
            out = data().long() if is_label else data().float()
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

        return out

    def __len__(self) -> int:
        """ Return the total number of samples in the dataset """
        return len(self._features)

    def __getitem__(self, index: Union[int, slice]) -> Union[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
        """ Return a single (feature, label) pair or a batch via slice """
        if isinstance(index, slice):
            # Return a batch (for example dataset[:5])
            return self._features[index], self._labels[index]
        elif isinstance(index, int):
            # Return a single sample
            return self._features[index], self._labels[index]
        else:
            raise TypeError(f"Invalid index type: {type(index)}")

    def __repr__(self):
        """ Return a string representation of the dataset """
        return f"LabelTorchDataset(features={self._features.shape}, labels={self._labels.shape}, device=cpu)"


class TorchDataLoader:
    """ A custom PyTorch DataLoader class for handling TorchDataset """

    def __init__(self, dataset: Dataset, batch_size: int = 32, is_shuffle: bool = True):
        """ Initialise the TorchDataLoader class
        :param dataset: the TorchDataset or Dataset to load data from
        :param batch_size: the number of samples per batch
        :param is_shuffle: whether to shuffle the data at every epoch
        """
        self._dataset: Union[Dataset, LabelTorchDataset] = dataset
        self._batches: int = batch_size
        self._is_shuffle: bool = is_shuffle

        self._loader: DataLoader = DataLoader(
            dataset=self._dataset,
            batch_size=self._batches,
            shuffle=self._is_shuffle,
        )

    @property
    def dataset(self) -> Union[Dataset, LabelTorchDataset]:
        return self._dataset

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """ Return a single (feature, label) pair or a batch via slice """
        if not isinstance(index, int):
            raise TypeError(f"Invalid index type: {type(index)}")
        return self._dataset[index]

    def __iter__(self):
        return iter(self._loader)

    def __len__(self) -> int:
        return len(self._loader)

    def __repr__(self):
        return (f"TorchDataLoader(dataset={self._dataset}, "
                f"batch_size={self._batches}, "
                f"shuffle={self._is_shuffle})")


class SequentialTorchDataset(Dataset):
    """ A custom PyTorch Dataset class for handling sequential features and labels """

    def __init__(self, sequences: list, sequence_length: int, pad_token: int) -> None:
        """ Initialise the TorchDataset class for sequential data
        :param sequences: the input sequences
        :param sequence_length: the length of each sequence
        :param pad_token: the padding token to use
        """
        self._sequences = sequences
        self._length = sequence_length
        self._pad = pad_token
        self._features, self._labels = self._pad_seq_to_tensor()

    def _pad_single_to_tensor(self) -> tuple[Tensor, Tensor]:
        """ Convert input data to a PyTorch tensor via padding
        :return: the converted PyTorch tensor
        """
        _features, _labels = [], []
        for i in range(len(self._sequences) - 1):
            if i < self._length - 1:
                feature = [self._pad] * (self._length - i - 1) + self._sequences[0: i + 1]
            else:
                feature = self._sequences[i - self._length + 1: i + 1]
            label = self._sequences[i + 1]
            _features.append(feature)
            _labels.append(label)

        return tensor(_features), tensor(_labels)

    def _pad_seq_to_tensor(self) -> tuple[Tensor, Tensor]:
        """ Convert input data to a PyTorch tensor via sequence padding
        :return: the converted PyTorch tensor
        """
        _features, _labels = [], []
        for i in range(len(self._sequences) - 1):
            if i < self._length - 1:
                feature = [self._pad] * (self._length - i - 1) + self._sequences[0: i + 1]
                label = [self._pad] * (self._length - i - 1) + self._sequences[1: i + 2]
            else:
                feature = self._sequences[i - self._length + 1: i + 1]
                label = self._sequences[i - self._length + 2: i + 2]

            _features.append(feature)
            _labels.append(label)

        return tensor(_features), tensor(_labels)

    def _slide_to_tensor(self) -> tuple[Tensor, Tensor]:
        """ Convert input data to a PyTorch tensor via sliding window
        :return: the converted PyTorch tensor
        """
        _features, _labels = [], []
        for i in range(len(self._sequences) - self._length):
            feature = self._sequences[i: i + self._length]
            label = self._sequences[i + self._length]
            _features.append(feature)
            _labels.append(label)

        return tensor(_features), tensor(_labels)

    @property
    def features(self) -> Tensor:
        """ Return the feature tensor as a property """
        return self._features

    @property
    def labels(self) -> Tensor:
        """ Return the label tensor as a property """
        return self._labels

    def __len__(self) -> int:
        """ Return the total number of samples in the dataset """
        return len(self._features)

    def __getitem__(self, index: Union[int, slice]) -> tuple[Tensor, Tensor]:
        """ Return a single (feature, label) pair or a batch via slice """
        if isinstance(index, slice):
            # Return a batch (for example dataset[:5])
            return self._features[index], self._labels[index]
        elif isinstance(index, int):
            # Return a single sample
            return self._features[index], self._labels[index]
        else:
            raise TypeError(f"Invalid index type: {type(index)}")

    def __repr__(self):
        """ Return a string representation of the dataset """
        return f"SequentialTorchDataset(features={self._features.shape}, labels={self._labels.shape}, device={self._features.device})"
