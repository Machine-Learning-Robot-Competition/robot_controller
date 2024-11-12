#!/usr/bin/env python3

import threading
import time
from PyQt5 import QtCore, QtGui, QtWidgets 
from python_qt_binding import loadUi
import sys
import rospy
import numpy as np
import logging
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
import pathlib
from typing import List
from controller_manager_msgs.srv import SwitchControllerRequest, SwitchController, LoadController, UnloadController, LoadControllerRequest, UnloadControllerRequest
from robot_controller.srv import GoForward, GoForwardRequest
import subprocess
import os
import toml
import pprint
from tf.transformations import quaternion_from_euler


UI_PATH = pathlib.Path(__file__).parent / "main.ui"
ROBOT_PACKAGE_NAME: str = "robot_controller"
ROBOT_NODE_NAME: str = "robot.py"
LAUNCH_NODE_CMD: List[str] = ['rosrun', ROBOT_PACKAGE_NAME, ROBOT_NODE_NAME]
CONFIG_PATH: str = str(pathlib.Path(__file__).absolute().parent.parent / "config" / "robot.toml")
CONTROLLERS: List[str] = [
    "controller/velocity",
    "controller/attitude"  # Yes, this is an intentional spelling of altitude as attitude
]
CONTROLLER_LOAD_SRV: str = "controller_manager/load_controller"
CONTROLLER_UNLOAD_SRV: str = "controller_manager/unload_controller"
CONTROLLER_UNLOAD_SRV: str = "controller_manager/unload_controller"
CONTROLLER_SWITCH_SRV: str = "controller_manager/switch_controller"
GO_FORWARD_SERVICE: str = "/go_forward"


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


with open(CONFIG_PATH) as f:
    robot_config = toml.load(f)

logger.info("Loaded Config: ")
logger.info(f"{pprint.pformat(robot_config)}")

initial_conditions: dict = robot_config["initial_conditions"]
robot_model_name: str = robot_config["info"]["model_name"]


def reset_robot_location():
    msg = ModelState()
    msg.model_name = robot_model_name

    orientation_quaternion = quaternion_from_euler(initial_conditions["R"], initial_conditions["P"], initial_conditions["Y"])

    msg.pose.position.x = initial_conditions["x"]
    msg.pose.position.y = initial_conditions["y"]
    msg.pose.position.z = initial_conditions["z"]
    msg.pose.orientation.x = orientation_quaternion[0]
    msg.pose.orientation.y = orientation_quaternion[1]
    msg.pose.orientation.z = orientation_quaternion[2]
    msg.pose.orientation.w = orientation_quaternion[3]
    msg.twist.linear.x = 0.0
    msg.twist.linear.y = 0.0
    msg.twist.linear.z = 0.0
    msg.twist.angular.x = 0.0
    msg.twist.angular.y = 0.0
    msg.twist.angular.z = 0.0

    print(msg.pose.position.x)
    print(type(msg.pose.position.x))

    rospy.wait_for_service('/gazebo/set_model_state')
    try:
        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        resp = set_state(msg)
        
        logger.info(f"Set Model State Service Call Succeeded: {resp}")

    except rospy.ServiceException as e:
        logger.error(f"Set Model State Service Call Failed: {e}")
    

class RobotUI(QtWidgets.QMainWindow):
    def __init__(self):
        super(RobotUI, self).__init__()
        loadUi(UI_PATH, self)

        self._node_process = None
        self._stdout_thread = None
        self._stderr_thread = None

        # Acquire services for controller loading and unloading 
        rospy.wait_for_service(CONTROLLER_LOAD_SRV)
        self._load_controller_service = rospy.ServiceProxy(CONTROLLER_LOAD_SRV,
                                              LoadController,
                                              persistent=True)
        
        rospy.wait_for_service(CONTROLLER_UNLOAD_SRV)
        self._unload_controller_service = rospy.ServiceProxy(CONTROLLER_UNLOAD_SRV,
                                              UnloadController,
                                              persistent=True)

        rospy.wait_for_service(CONTROLLER_SWITCH_SRV)
        self._switch_controller_service = rospy.ServiceProxy(CONTROLLER_SWITCH_SRV,
                                              SwitchController,
                                              persistent=True)
        
        self.begin_button.clicked.connect(self.SLOT_begin_button)
        self.reset_drone.clicked.connect(self.SLOT_reset_drone)
        self.reset_model.clicked.connect(self.SLOT_reset_model)
        self.go_forward_button.clicked.connect(self.SLOT_go_forward)

    def _launch_node(self):
        """
        Try to launch the ROS Robot Node, if it doesn't already exist.
        """
        if self._node_process is None or self._node_process.poll() is not None:
            self._node_process = subprocess.Popen(
                LAUNCH_NODE_CMD,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=dict(os.environ, PYTHONUNBUFFERED="1")
            )

            # We will start two threads that will read the node process's stdout and stderr and log it
            def stream_stdout():
                """Thread function to read and print stdout line by line."""
                for line in self._node_process.stdout:
                    logger.info(line)

            def stream_stderr():
                """Thread function to read and print stderr line by line."""
                for line in self._node_process.stdout:
                    logger.error(line)

            self._stdout_thread = threading.Thread(target=stream_stdout)
            self._stderr_thread = threading.Thread(target=stream_stderr)

            self._stdout_thread.start()
            self._stderr_thread.start()

            logger.info("Robot Node has been started.")
            
        else:
            logger.warning("Cannot start Robot Node: already started!")

    def _kill_node(self):
        """
        Stop the ROS Robot Node, if it is running.
        """
        if self._node_process and self._node_process.poll() is None:
            self._node_process.terminate()

            # Kill the logging threads
            self._stderr_thread.join()
            self._stdout_thread.join()

            self._node_process.wait()

            logger.info("Robot Node has been stopped.")

        else:
            logger.warning("Cannot stop Robot Node: not started!")

    def _go_forward(self):
        """
        Initiate a request for the drone to perform a "go forward" operation.
        """
        rospy.wait_for_service(GO_FORWARD_SERVICE)
        try:
            go_forward = rospy.ServiceProxy(GO_FORWARD_SERVICE, GoForward)

            request = GoForwardRequest()

            response = go_forward(request)

            logger.info(f"Go Forward Response {response}")

        except rospy.ServiceException as e:
            logger.error(f"Failed to go forward: \n {e}")

    def SLOT_begin_button(self):
        logger.info("Trying to begin ROS Node...")
        self._launch_node()

    def SLOT_reset_drone(self):
        logger.info("Trying to reset drone...")
        self._kill_node()

    def SLOT_reset_model(self):
        logger.info("Trying to Reset Model...")

        self._unload_controllers()  # Stop the controllers to try to avoid weird PID stuff

        reset_robot_location()

        self._load_controllers()  # Restart controllers after a brief pause to try to avoid weird PID stuff

        logger.info("Model Reset!")

    def SLOT_go_forward(self):
        logger.info("Trying to go forward...")

        self._go_forward()

    def _load_controllers(self):
        """
        Tries to load all of the controllers defined in `CONTROLLERS`.

        Adapted from https://github.com/ros-controls/ros_control/blob/noetic-devel/rqt_controller_manager/src/rqt_controller_manager/controller_manager.py
        """
        for controller in CONTROLLERS:
            try:
                request = LoadControllerRequest()
                request.name = controller

                response = self._load_controller_service(request)

                logger.info(f"Service Call Response to Load {controller}: {response}")

                # Now, start the controller
                switch_requqest = SwitchControllerRequest(start_controllers=[controller],
                                                          stop_controllers=[],
                                                          strictness=SwitchControllerRequest.STRICT)
                
                try:
                    switch_response = self._switch_controller_service(switch_requqest)

                    logger.info(f"Service Call Response to Start {controller}: {switch_response}")

                except rospy.ServiceException as e:
                        logger.error(f"Failed to Start Controller {controller}:\n {e}")

            except rospy.ServiceException as e:
                logger.error(f"Failed to Load Controller {controller}:\n {e}")
                break

    def _unload_controllers(self):
        """
        Tries to unload all of the controllers defined in `CONTROLLERS`.

        Adapted from https://github.com/ros-controls/ros_control/blob/noetic-devel/rqt_controller_manager/src/rqt_controller_manager/controller_manager.py
        """
        for controller in CONTROLLERS:
            try:
                # First, we need to stop the controller
                switch_requqest = SwitchControllerRequest(start_controllers=[],
                                                          stop_controllers=[controller],
                                                          strictness=SwitchControllerRequest.STRICT)
                
                try:
                    switch_response = self._switch_controller_service(switch_requqest)

                    logger.info(f"Service Call Response to Stop {controller}: {switch_response}")

                except rospy.ServiceException as e:
                        logger.error(f"Failed to Stop Controller {controller}:\n {e}")
                        continue  # We cannot continue with this controller if we couldn't stop it 
                
                stop_request = UnloadControllerRequest()
                stop_request.name = controller

                stop_response = self._unload_controller_service(stop_request)

                logger.info(f"Service Call Response to Unload {controller}: {stop_response}")

            except rospy.ServiceException as e:
                logger.error(f"Failed to Unload Controller {controller}:\n {e}")
                break


if __name__ == "__main__":
	app = QtWidgets.QApplication(sys.argv)
	sift_demo_app = RobotUI()
	sift_demo_app.show()
	sys.exit(app.exec_())
