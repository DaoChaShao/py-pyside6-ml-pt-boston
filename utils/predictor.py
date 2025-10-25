#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/25 14:00
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   predictor.py
# @Desc     :   


from pathlib import Path
from random import randint
from torch import load

from main import prepare_data
from utils.config import CONFIG
from utils.helper import Timer
from utils.highlighter import red, green
from utils.models import RegressionTorchModel


def main() -> None:
    """ Main Function """
    with Timer("Data Preparation"):
        path = Path(CONFIG.FILEPATHS.MODEL)
        if path.exists():
            print("The model file exists.")

            _, loader = prepare_data()
            index: int = randint(0, len(loader) - 1)
            # print(loader[index])

            features: int = loader[0][0].shape[0]
            model = RegressionTorchModel(
                features,
                CONFIG.PARAMETERS.FC_HIDDEN_UNITS,
                1,
                CONFIG.PARAMETERS.FC_DROPOUT_RATE
            )
            state_dict = load(CONFIG.FILEPATHS.MODEL)
            model.load_state_dict(state_dict)
            model.eval()

            out = model(loader[index][0].unsqueeze(0))
            print(f"Predicted value: {out.item():.4f}, Actual value: {loader[index][1].item():.4f}")
            print(green("Successfully!") if abs(out.item() - loader[index][1].item()) < 3 else red("Unsuccessfully!"))
        else:
            print("The model file does not exist.")


if __name__ == "__main__":
    main()
