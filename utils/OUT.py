#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/25 00:36
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   OUT.py
# @Desc     :   

from dataclasses import dataclass


@dataclass
class Outputter:
    """ Print Class with ON/OFF switch """
    _enabled: bool = True

    def yes(self):
        """ Open output """
        self._enabled = True

    def no(self):
        """ Close output """
        self._enabled = False

    def print(self, msg: str) -> None:
        """ Display the message if switch is on """
        if self._enabled:
            print(msg)


out = Outputter()
