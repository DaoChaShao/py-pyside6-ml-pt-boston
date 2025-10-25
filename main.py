#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/25 00:19
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   main.py
# @Desc     :

from pandas import DataFrame
from random import randint
from torch import optim, nn

from utils.config import CONFIG
from utils.helper import load_text_data, Timer
from utils.models import RegressionTorchModel
from utils.PT import LabelTorchDataset, TorchDataLoader
from utils.stats import split_data, preprocess_data as preprocess_data_stats
from utils.trainer import TorchTrainer


def preprocess_data():
    """ Data Preprocessing Function """
    # Get the dataset
    columns: dict[str, str] = {
        "CRIM": "城镇人均犯罪率",
        "ZN": "住宅用地超过 25,000 平方英尺的比例",
        "INDUS": "城镇中非零售商用土地所占比例",
        "CHAS": "查尔斯河虚拟变量（1 表示临河；0 表示不临河）",
        "NOX": "一氧化氮浓度（每 1000 万分之一）",
        "RM": "每栋住宅的平均房间数",
        "AGE": "1940 年前建成的自有住房比例",
        "DIS": "距离波士顿五个中心区域的加权距离",
        "RAD": "辐射公路可达性指数",
        "TAX": "每一万美元的房产税率",
        "PTRATIO": "城镇师生比例",
        "B": "1000(Bk - 0.63)^2，其中 Bk 为城镇中黑人的比例",
        "LSTAT": "低收入人群所占比例（%）",
        "MEDV": "自住房的房价中位数（千美元）"
    }
    cols: list[str] = [key for key, _ in columns.items()]
    # print(cols)
    data = load_text_data(CONFIG.FILEPATHS.BOSTON_HOUSE_PRICES, True, cols)
    # print(data.head())
    # Get X and y
    X = data.iloc[:, :-1]
    # print(X.head(), X.shape)
    y = data.iloc[:, -1]
    # print(y.head(), y.shape)
    y = DataFrame(y)
    # print(X)

    X_train, X_test, y_train, y_test = split_data(X, y)
    # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    return X_train, X_test, y_train, y_test


def prepare_data():
    """ Data Preparation Function """
    X_train, X_test, y_train, y_test = preprocess_data()

    X_train, preprocessor = preprocess_data_stats(X_train)
    X_test = preprocessor.transform(X_test)

    # Create Torch Datasets
    train_dataset = LabelTorchDataset(X_train, y_train)
    test_dataset = LabelTorchDataset(X_test, y_test)

    # Create DataLoaders
    train_loader = TorchDataLoader(
        train_dataset,
        batch_size=CONFIG.PREPROCESSOR.BATCH_SIZE,
        is_shuffle=CONFIG.PREPROCESSOR.IS_SHUFFLE,
    )
    test_loader = TorchDataLoader(
        test_dataset,
        batch_size=CONFIG.PREPROCESSOR.BATCH_SIZE,
        is_shuffle=CONFIG.PREPROCESSOR.IS_SHUFFLE,
    )

    return train_loader, test_loader


def main() -> None:
    """ Main Function """
    train_loader, test_loader = prepare_data()
    # index_train: int = randint(0, len(train_loader) - 1)
    # index_test: int = randint(0, len(test_loader) - 1)
    # print(index_train, index_test)
    # print(train_loader[index_train], test_loader[index_test])

    # Initialise a linear model
    features: int = train_loader[0][0].shape[0]
    # print(features)
    out_features: int = 1
    model = RegressionTorchModel(features, CONFIG.PARAMETERS.FC_HIDDEN_UNITS, 1, CONFIG.PARAMETERS.FC_DROPOUT_RATE)
    model.summary()

    # Setup optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=CONFIG.HYPERPARAMETERS.ALPHA)
    criterion = nn.MSELoss()

    # Initialise Trainer
    trainer = TorchTrainer(model, optimizer, criterion, CONFIG.HYPERPARAMETERS.ACCELERATOR)
    # Train the model
    trainer.fit(train_loader, test_loader, CONFIG.HYPERPARAMETERS.EPOCHS, CONFIG.FILEPATHS.MODEL)


if __name__ == "__main__":
    main()
