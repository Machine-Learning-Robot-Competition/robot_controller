#!/usr/bin/env python3

from PyQt5 import QtCore, QtGui, QtWidgets 
from python_qt_binding import loadUi

import sys
import numpy as np
import logging
import pathlib
from typing import List
import subprocess


UI_PATH = pathlib.Path(__file__).parent / "main.ui"
ROBOT_PACKAGE_NAME: str = "robot_controller"
ROBOT_NODE_NAME: str = "robot.py"
LAUNCH_NODE_CMD: List[str] = ['rosrun', ROBOT_PACKAGE_NAME, ROBOT_NODE_NAME]


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class RobotUI(QtWidgets.QMainWindow):
    def __init__(self):
        super(RobotUI, self).__init__()
        loadUi(UI_PATH, self)

        self._node_process = None

        self.begin_button.clicked.connect(self.SLOT_begin_button)
        self.reset_drone.clicked.connect(self.SLOT_reset_drone)
        self.reset_model.clicked.connect(self.SLOT_reset_model)

    def _launch_node(self):
        """
        Try to launch the ROS Robot Node, if it doesn't already exist.
        """
        if self._node_process is None or self._node_process.poll() is not None:
            self._node_process = subprocess.Popen(
                LAUNCH_NODE_CMD,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )

            logger.info("Robot Node has been started.")
            
        else:
            logger.warning("Cannot start Robot Node: already started!")

    def _kill_node(self):
        """
        Stop the ROS Robot Node, if it is running.
        """
        if self._node_process and self._node_process.poll() is None:
            self._node_process.terminate()
            self._node_process.wait()  # Ensure the process has terminated

            logger.info("Robot Node has been stopped.")

        else:
            logger.warning("Cannot stop Robot Node: not started!")


    def SLOT_begin_button(self):
        logger.info("Trying to begin ROS Node...")
        self._launch_node()

    def SLOT_reset_drone(self):
        logger.info("Trying to reset drone...")
        self._kill_node()

    def SLOT_reset_model(self):
        logger.info("Reset Model!")


if __name__ == "__main__":
	app = QtWidgets.QApplication(sys.argv)
	sift_demo_app = RobotUI()
	sift_demo_app.show()
	sys.exit(app.exec_())
