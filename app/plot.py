#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/25 16:55
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   plot.py
# @Desc     :   

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtCharts import QLineSeries, QChart, QChartView
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget,
                               QVBoxLayout, QHBoxLayout,
                               QPushButton)
from sys import argv, exit

from torch import optim, nn

from main import prepare_data
from utils.config import CONFIG
from utils.models import RegressionTorchModel
from utils.trainer import TorchTrainer


class Train(QThread):
    losses = Signal(int, float, float)

    def __init__(self):
        super().__init__()
        self._train, self._test = train_loader, test_loader = prepare_data()
        self._model = RegressionTorchModel(
            self._train[0][0].shape[0],
            CONFIG.PARAMETERS.FC_HIDDEN_UNITS,
            1,
            CONFIG.PARAMETERS.FC_DROPOUT_RATE
        )
        self._optimiser = optim.Adam(self._model.parameters(), lr=CONFIG.HYPERPARAMETERS.ALPHA)
        self._criterion = nn.MSELoss()

    def run(self):
        trainer = TorchTrainer(self._model, self._optimiser, self._criterion, CONFIG.HYPERPARAMETERS.ACCELERATOR)
        trainer.processor.connect(self.losses)
        trainer.fit(self._train, self._test, CONFIG.HYPERPARAMETERS.EPOCHS, CONFIG.FILEPATHS.MODEL)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Boston Housing Prices Losses Plot")
        self.resize(800, 400)
        self._widget = QWidget(self)
        self.setCentralWidget(self._widget)

        self._btn_names = ["Plot", "Clear", "Exit"]
        self._buttons = []
        self._chart = QChart()
        self._view = QChartView(self._chart)
        self._legends = ["Train Loss", "Test Loss"]
        self._series = []

        self._setup()

        self._thread = Train()
        self._thread.losses.connect(self._get_losses)

    def _setup(self):
        # Main widget and layout
        _layout = QVBoxLayout(self._widget)
        _row_btn = QHBoxLayout()

        self._view.setRenderHint(QPainter.RenderHint.Antialiasing)
        _layout.addWidget(self._view)

        funcs = [
            self._plot,
            self._clear,
            self._exit
        ]
        for i, name in enumerate(self._btn_names):
            btn = QPushButton(name)
            btn.clicked.connect(funcs[i])
            _row_btn.addWidget(btn)
            self._buttons.append(btn)
            match btn.text():
                case "Clear":
                    btn.setEnabled(False)
        _layout.addLayout(_row_btn)

    def _plot(self):
        # Initialise the chart and data docker
        self._chart.removeAllSeries()

        for i in self._legends:
            series = QLineSeries()
            series.setName(i)
            self._series.append(series)
            self._chart.addSeries(series)
        self._chart.createDefaultAxes()
        self._chart.setTitle(" & ".join(self._btn_names))

        # Start the training thread
        self._thread.start()

        for btn in self._buttons:
            match btn.text():
                case "Clear":
                    btn.setEnabled(True)
                case "Plot":
                    btn.setEnabled(False)

    def _clear(self):
        self._chart.setTitle("")
        self._chart.removeAllSeries()
        for axis in self._chart.axes():
            self._chart.removeAxis(axis)

        for btn in self._buttons:
            match btn.text():
                case "Clear":
                    btn.setEnabled(False)
                case "Plot":
                    btn.setEnabled(True)

    def _exit(self):
        self.close()

    def _get_losses(self, epoch: int, train_loss: float, test_loss: float) -> None:
        """ Get losses from the training thread and update the plot """
        self._series[0].append(epoch, train_loss)
        self._series[1].append(epoch, test_loss)

        # Set axis x range dynamically
        axis_x = self._chart.axes(Qt.Orientation.Horizontal, self._series[0])[0]
        axis_x.setRange(0, epoch * 1.1)

        # Set axis y range dynamically
        axis_y = self._chart.axes(Qt.Orientation.Vertical, self._series[0])[0]
        points_y = [self._series[0].at(i).y() for i in range(self._series[0].count())]
        if points_y:
            min_y = min(points_y)
            max_y = max(points_y)
            axis_y.setRange(min_y * 0.9, max_y * 1.1)

        # Update the view
        self._view.update()


def main() -> None:
    """ Main Function """
    app = QApplication(argv)
    window = MainWindow()
    window.show()
    exit(app.exec())


if __name__ == "__main__":
    main()
