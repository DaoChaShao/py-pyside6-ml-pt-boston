#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/25 16:16
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   table.py
# @Desc     :   

from PySide6.QtCore import QSortFilterProxyModel
from PySide6.QtGui import QStandardItemModel, QStandardItem
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget,
                               QVBoxLayout, QHBoxLayout,
                               QPushButton,
                               QTableView, QAbstractItemView)
from sys import argv, exit

from utils.config import CONFIG
from utils.helper import load_text_data


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Boston Housing Prices Table")
        self.resize(800, 600)
        self._widget = QWidget(self)
        self.setCentralWidget(self._widget)

        self._btn_names = ["Load", "Clear", "Exit"]
        self._buttons = []
        self._table = QTableView()
        self._model = QStandardItemModel()
        self._proxy = QSortFilterProxyModel()

        self._setup()

    def _setup(self):
        # Main widget and layout
        _layout = QVBoxLayout(self._widget)
        _row_btn = QHBoxLayout()

        _layout.addWidget(self._table)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSortingEnabled(True)

        funcs = [
            self._load,
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

    def _load(self):
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
        data = load_text_data(CONFIG.FILEPATHS.BOSTON_HOUSE_PRICES, True, cols)
        # print(data.head())

        for row_idx, row in data.iterrows():
            items = [QStandardItem(str(row[col])) for col in cols]
            self._model.appendRow(items)
        self._model.setHorizontalHeaderLabels(cols)
        self._proxy.setSourceModel(self._model)
        self._table.setModel(self._proxy)

        for btn in self._buttons:
            match btn.text():
                case "Load":
                    btn.setEnabled(False)
                case "Clear":
                    btn.setEnabled(True)

    def _clear(self):
        self._model.clear()

        for btn in self._buttons:
            match btn.text():
                case "Load":
                    btn.setEnabled(True)
                case "Clear":
                    btn.setEnabled(False)

    def _exit(self):
        self.close()


def main() -> None:
    """ Main Function """
    app = QApplication(argv)
    window = MainWindow()
    window.show()
    exit(app.exec())


if __name__ == "__main__":
    main()
