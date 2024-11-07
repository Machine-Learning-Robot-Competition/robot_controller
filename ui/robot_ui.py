#!/usr/bin/env python3

from PyQt5 import QtCore, QtGui, QtWidgets
from python_qt_binding import loadUi

import cv2
import sys
import numpy as np
import logging
import pathlib


UI_PATH = pathlib.Path(__file__).parent / "main.ui"


class RobotUI(QtWidgets.QMainWindow):
    def __init__(self):
        super(RobotUI, self).__init__()
        loadUi(UI_PATH, self)


if __name__ == "__main__":
	app = QtWidgets.QApplication(sys.argv)
	sift_demo_app = RobotUI()
	sift_demo_app.show()
	sys.exit(app.exec_())
