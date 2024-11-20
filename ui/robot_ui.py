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
from std_msgs.msg import Int16
from geometry_msgs.msg import Twist
import pathlib
from typing import List
from controller_manager_msgs.srv import SwitchControllerRequest, SwitchController, LoadController, UnloadController, LoadControllerRequest, UnloadControllerRequest
from robot_controller.srv import GoForward, GoForwardRequest
import subprocess
import os
import toml
import pprint
from tf.transformations import quaternion_from_euler
from cv_utils import convert_cv_to_pixmap
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import traceback


# File Paths
UI_PATH = pathlib.Path(__file__).parent / "main.ui"
CONFIG_PATH: str = str(pathlib.Path(__file__).absolute().parent.parent / "config" / "robot.toml")

# Topic and Service Names
CONTROLLER_LOAD_SRV: str = "controller_manager/load_controller"
CONTROLLER_UNLOAD_SRV: str = "controller_manager/unload_controller"
CONTROLLER_UNLOAD_SRV: str = "controller_manager/unload_controller"
CONTROLLER_SWITCH_SRV: str = "controller_manager/switch_controller"
GO_FORWARD_SERVICE: str = "/go_forward"
VISION_TOPIC: str = "/robot_brain/vision"
SET_MODEL_STATE_SERVICE: str = "/gazebo/set_model_state" 
ROBOT_COMMAND_TOPIC = "/robot_state_command"
CAMERA_FEED_TOPIC: str = "/front_cam/camera/image"
NUM_GOOD_POINTS_TOPIC: str = "/robot_brain/good_points"
EXTRACTED_IMAGE_TOPIC: str = "/robot_brain/extracted_image"

# Other Declarations
ROBOT_PACKAGE_NAME: str = "robot_controller"
ROBOT_CONTROL_NODE_NAME: str = "robot_control.py"
ROBOT_BRAIN_NODE_NAME: str = "robot_brain.py"
LAUNCH_CONTROL_NODE_CMD: List[str] = ['rosrun', ROBOT_PACKAGE_NAME, ROBOT_CONTROL_NODE_NAME]
LAUNCH_BRAIN_NODE_CMD: List[str] = ['rosrun', ROBOT_PACKAGE_NAME, ROBOT_BRAIN_NODE_NAME]
CONTROLLERS: List[str] = ["controller/velocity", "controller/attitude"]
TIME_ELAPSED_UPDATE_PERIOD = 10 # ms


with open(CONFIG_PATH) as f:
    robot_config = toml.load(f)

rospy.loginfo("Loaded Config: ")
rospy.loginfo(f"{pprint.pformat(robot_config)}")

initial_conditions: dict = robot_config["initial_conditions"]
robot_model_name: str = robot_config["info"]["model_name"]


class RobotUI(QtWidgets.QMainWindow):
    def __init__(self):
        super(RobotUI, self).__init__()
        loadUi(UI_PATH, self)

        self.begin_button.clicked.connect(self.SLOT_begin_button)
        self.control_state.clicked.connect(self.SLOT_control_state)
        self.brain_state.clicked.connect(self.SLOT_brain_state)
        self.reset_model.clicked.connect(self.SLOT_reset_model)
        self.go_forward_button.clicked.connect(self.SLOT_go_forward)

        # We want to run our robot brain and control nodes, so we will start two new processes for each
        # Each will also have two threads to capture their stdout and stderr and pipe it back to this main process
        self._control_node_process: subprocess.Popen = None
        self._control_stdout_thread: threading.Thread = None
        self._control_stderr_thread: threading.Thread = None
        self._brain_node_process: subprocess.Popen = None
        self._brain_stdout_thread: threading.Thread = None
        self._brain_stderr_thread: threading.Thread = None
                
        # We can use controller services to load and unload controllers
        # Controllers are what the drone uses to control itself. We need to unload and reload them when resetting the model position
        # or the PID tries to self-immolate. Also, when we want it to reset we just in general want everything to reset.
        rospy.wait_for_service(CONTROLLER_LOAD_SRV)
        self._load_controller_service = rospy.ServiceProxy(CONTROLLER_LOAD_SRV, LoadController, persistent=True)
        
        rospy.wait_for_service(CONTROLLER_UNLOAD_SRV)
        self._unload_controller_service = rospy.ServiceProxy(CONTROLLER_UNLOAD_SRV, UnloadController, persistent=True)

        rospy.wait_for_service(CONTROLLER_SWITCH_SRV)
        self._switch_controller_service = rospy.ServiceProxy(CONTROLLER_SWITCH_SRV, SwitchController, persistent=True)

        rospy.wait_for_service(SET_MODEL_STATE_SERVICE)
        self._set_model_state = rospy.ServiceProxy(SET_MODEL_STATE_SERVICE, SetModelState, persistent=True)

        # Publisher to publish commands to the robot command topic (robot_control reads this topic, and updates cmd_vel with it continuously)
        self._command_publisher = rospy.Publisher(ROBOT_COMMAND_TOPIC, Twist, queue_size=1)
        self._going_forward = False

        # OpenCV stuff
        self._cv_bridge = CvBridge()
        self._vision_subscriber = rospy.Subscriber(VISION_TOPIC, Image, self._read_vision)
        self._camera_feed_subscriber = rospy.Subscriber(CAMERA_FEED_TOPIC, Image, self._update_camera_feed)
        self._extracted_image_subscriber = rospy.Subscriber(EXTRACTED_IMAGE_TOPIC, Image, self._update_extracted_image)

        self._num_good_points = 0
        self._num_good_points_subscriber = rospy.Subscriber(NUM_GOOD_POINTS_TOPIC, Int16, self._update_good_points)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._update_elapsed_time)
        self.timer.start(TIME_ELAPSED_UPDATE_PERIOD)  # Update every second
        self._time_since_update = 0
                
    def _update_elapsed_time(self):
        self._time_since_update += TIME_ELAPSED_UPDATE_PERIOD
        self.update_label.setText(f"Time Elapsed since Update: {self._time_since_update}ms") 

    def _read_vision(self, msg):
        """
        Callback for the robot vision topic to update our UI with what the robot is seeing
        """
        try:
            cv_image: np.ndarray = self._cv_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            pixmap = convert_cv_to_pixmap(cv_image)
            scaled_pixmap = pixmap.scaled(self.vision_label.size(), aspectRatioMode=True)
            self.vision_label.setPixmap(scaled_pixmap)

        except Exception as e:
            rospy.logerr(f"{e}: {traceback.format_exc()}")

    def _update_camera_feed(self, msg):
        """
        Callback for the camera feed topic to update our UI with what the camera is seeing
        """
        try:
            cv_image: np.ndarray = self._cv_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            pixmap = convert_cv_to_pixmap(cv_image)
            scaled_pixmap = pixmap.scaled(self.camera_feed.size(), aspectRatioMode=True)
            self.camera_feed.setPixmap(scaled_pixmap)

        except Exception as e:
            rospy.logerr(f"{e}: {traceback.format_exc()}")

    def _update_good_points(self, num_good_points: Int16):
        """
        Update the number of good points that are being detected by SIFT
        """
        self.homography_status_text.setText(f"Number of good points identified: {num_good_points.data}")

    def _update_extracted_image(self, msg: Image):
        try:
            cv_image: np.ndarray = self._cv_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            pixmap = convert_cv_to_pixmap(cv_image)
            scaled_pixmap = pixmap.scaled(self.sign_label.size(), aspectRatioMode=True)
            self.sign_label.setPixmap(scaled_pixmap)
            self._time_since_update = 0.0

        except Exception as e:
            rospy.logerr(f"{e}: {traceback.format_exc()}")
            
    def _reset_robot_location(self):
        """
        Reset the position of the robot model to the starting position. Also, zero the angular and linear velocity.
        """
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

        try:
            resp = self._set_model_state(msg)
            
            rospy.loginfo(f"Set Model State Service Call Succeeded: {resp}")

        except rospy.ServiceException as e:
            rospy.logerr(f"Set Model State Service Call Failed: {e}")

    def _launch_nodes(self):
        """
        Launch the ROS Robot Nodes
        """
        
        self._enable_brain()
        self._enable_control()

        self.begin_button.setEnabled(False)

    def _enable_brain(self):
        self._brain_node_process, self._brain_stdout_thread, self._brain_stderr_thread = self._launch_node(self._brain_node_process, LAUNCH_BRAIN_NODE_CMD)
        self.brain_state.setText("Kill Brain")

    def _enable_control(self):
        self._control_node_process, self._control_stdout_thread, self._control_stderr_thread = self._launch_node(self._control_node_process, LAUNCH_CONTROL_NODE_CMD)
        self.control_state.setText("Kill Control")

    def _kill_control(self):
        self._kill_node(self._control_node_process, self._control_stdout_thread, self._control_stderr_thread)
        self.control_state.setText("Enable Control")
    
    def _kill_brain(self):
        self._kill_node(self._brain_node_process, self._brain_stdout_thread, self._brain_stderr_thread)
        self.brain_state.setText("Enable Brain")


    def _launch_node(self, process: subprocess.Popen, cmd):
        """
        Try to launch a node, if it doesn't already exist.
        """
        if process is None or process.poll() is not None:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=dict(os.environ, PYTHONUNBUFFERED="1")
            )

            # We will start two threads that will read the node process's stdout and stderr and log it
            def stream_stdout():
                """Thread function to read and print stdout line by line."""
                for line in process.stdout:
                    rospy.loginfo(line)

            def stream_stderr():
                """Thread function to read and print stderr line by line."""
                for line in process.stderr:
                    rospy.logerr(line)

            _stdout_thread = threading.Thread(target=stream_stdout)
            _stderr_thread = threading.Thread(target=stream_stderr)

            _stdout_thread.start()
            _stderr_thread.start()

            rospy.loginfo("Robot Node has been started.")

            return process, _stdout_thread, _stderr_thread
            
        else:
            rospy.logerr("Cannot start Robot Node: already started!")

    def _kill_node(self, process: subprocess.Popen, stdout_thread: threading.Thread, stderr_thread: threading.Thread):
        """
        Stop a node, if it is running.
        """
        if process and process.poll() is None:
            process.terminate()

            # Kill the logging threads
            stderr_thread.join()
            stdout_thread.join()

            process.wait()

            rospy.loginfo("Robot Node has been stopped.")

        else:
            rospy.logerr("Cannot stop Robot Node: not started!")

    def _go_forward(self):
        """
        Initiate a request for the drone to perform a "go forward" operation.
        """
        twist = Twist()

        if self._going_forward:
            twist.linear.x = 0.0 
            self._going_forward = False
        else:
            twist.linear.x = 1.0 
            self._going_forward = True

        self._command_publisher.publish(twist)
        # rospy.wait_for_service(GO_FORWARD_SERVICE)
        # try:
        #     go_forward = rospy.ServiceProxy(GO_FORWARD_SERVICE, GoForward)

        #     request = GoForwardRequest()

        #     response = go_forward(request)

        #     rospy.loginfo(f"Go Forward Response {response}")

        # except rospy.ServiceException as e:
        #     rospy.logerror(f"Failed to go forward: \n {e}")

    def SLOT_begin_button(self):
        rospy.loginfo("Trying to begin ROS Node...")
        self._launch_nodes()

    def SLOT_control_state(self):
        rospy.loginfo("Switching control state...")

        if self.control_state.text() == "Enable Control":
            self._enable_control()
            rospy.loginfo("Enabled control!")

        else:
            self._kill_control()
            rospy.loginfo("Killed control!")

    def SLOT_brain_state(self):
        rospy.loginfo("Switching brain state...")
        
        if self.brain_state.text() == "Enable Brain":
            self._enable_brain()
            rospy.loginfo("Enabled control!")

        else:
            self._kill_brain()
            rospy.loginfo("Killed control!")

    def SLOT_reset_model(self):
        rospy.loginfo("Trying to Reset Model...")

        self._unload_controllers()  # Stop the controllers to try to avoid weird PID stuff

        self._reset_robot_location()

        self._load_controllers()  # Restart controllers after a brief pause to try to avoid weird PID stuff

        rospy.loginfo("Model Reset!")

    def SLOT_go_forward(self):
        rospy.loginfo("Trying to go forward...")

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

                rospy.loginfo(f"Service Call Response to Load {controller}: {response}")

                # Now, start the controller
                switch_requqest = SwitchControllerRequest(start_controllers=[controller],
                                                          stop_controllers=[],
                                                          strictness=SwitchControllerRequest.STRICT)
                
                try:
                    switch_response = self._switch_controller_service(switch_requqest)

                    rospy.loginfo(f"Service Call Response to Start {controller}: {switch_response}")

                except rospy.ServiceException as e:
                        rospy.logerr(f"Failed to Start Controller {controller}:\n {e}")

            except rospy.ServiceException as e:
                rospy.logerr(f"Failed to Load Controller {controller}:\n {e}")
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

                    rospy.loginfo(f"Service Call Response to Stop {controller}: {switch_response}")

                except rospy.ServiceException as e:
                        rospy.logerr(f"Failed to Stop Controller {controller}:\n {e}")
                        continue  # We cannot continue with this controller if we couldn't stop it 
                
                stop_request = UnloadControllerRequest()
                stop_request.name = controller

                stop_response = self._unload_controller_service(stop_request)

                rospy.loginfo(f"Service Call Response to Unload {controller}: {stop_response}")

            except rospy.ServiceException as e:
                rospy.logerr(f"Failed to Unload Controller {controller}:\n {e}")
                break


if __name__ == "__main__":
    rospy.init_node('robot_ui')
    time.sleep(1.0)  # Wait 1.0s for node to be registered

    app = QtWidgets.QApplication(sys.argv)
    sift_demo_app = RobotUI()
    sift_demo_app.show()
    sys.exit(app.exec_())
