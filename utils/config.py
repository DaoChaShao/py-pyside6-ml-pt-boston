#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/25 00:18
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   config.py
# @Desc     :   

from dataclasses import dataclass, field
from pathlib import Path
from torch import cuda

BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass
class FilePaths:
    MODEL: Path = BASE_DIR / "models/model.pth"
    BOSTON_HOUSE_PRICES = BASE_DIR / "data/boston_house_prices.txt"


@dataclass
class DataPreprocessor:
    RANDOM_STATE: int = 27
    VALID_SIZE: float = 0.2
    IS_SHUFFLE: bool = True
    PCA_VARIANCE_THRESHOLD: float = 0.95
    BATCH_SIZE: int = 32


@dataclass
class ModelParameters:
    FC_HIDDEN_UNITS: int = 128
    FC_DROPOUT_RATE: float = 0.2
    RNN_SEQUENTIAL_LENGTH: int = 12
    RNN_EMBEDDING_DIM: int = 256
    RNN_HIDDEN_SIZE: int = 512
    RNN_LAYERS: int = 2
    RNN_TEMPERATURE: float = 1.0


@dataclass
class Hyperparameters:
    ALPHA: float = 1e-3
    ALPHA_REDUCTION: float = 0.3
    EPOCHS: int = 20
    ACCELERATOR: str = "cuda" if cuda.is_available() else "cpu"


@dataclass
class Configration:
    FILEPATHS: FilePaths = field(default_factory=FilePaths)
    PREPROCESSOR: DataPreprocessor = field(default_factory=DataPreprocessor)
    PARAMETERS: ModelParameters = field(default_factory=ModelParameters)
    HYPERPARAMETERS: Hyperparameters = field(default_factory=Hyperparameters)


CONFIG = Configration()
