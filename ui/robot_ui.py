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
from std_msgs.msg import Int16, Bool
from geometry_msgs.msg import Twist
import pathlib
from typing import List
from controller_manager_msgs.srv import SwitchControllerRequest, SwitchController, LoadController, UnloadController, LoadControllerRequest, UnloadControllerRequest
from robot_controller.srv import GoForward, GoForwardRequest, GoForwardResponse
import subprocess
import os
import toml
import pprint
from tf.transformations import quaternion_from_euler
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import traceback
import actionlib
from node_utils import NodeThread
from robot_controller.msg import ReadClueboardAction, ReadClueboardGoal, ReadClueboardFeedback, ReadClueboardResult
import cv2


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
LETTERS_PUBLISH_TOPIC: str = "/robot_brain/letters_image"

# Other Declarations
ROBOT_PACKAGE_NAME: str = "robot_controller"
ROBOT_CONTROL_NODE_NAME: str = "robot_control.py"
ROBOT_BRAIN_NODE_NAME: str = "robot_brain.py"
NAVIGATION_CONTROLLER_NAME: str = "navigation_controller.py"
MASTER_CONTROLLER_NAME: str = "master_control.py"
CONTROLLERS: List[str] = ["controller/velocity", "controller/attitude"]
TIME_ELAPSED_UPDATE_PERIOD = 10 # ms


with open(CONFIG_PATH) as f:
    robot_config = toml.load(f)

rospy.loginfo("Loaded Config: ")
rospy.loginfo(f"{pprint.pformat(robot_config)}")

initial_conditions: dict = robot_config["initial_conditions"]
robot_model_name: str = robot_config["info"]["model_name"]


# Source: stackoverflow.com/questions/34232632/
def convert_cv_to_pixmap(cv_img: np.ndarray) -> QtGui.QPixmap:
    """
    Convert an OpenCV-compatible ndarray to a Qt-compatible Pixmap.

    @param np.ndarray cv_img: OpenCV-compatible ndarray that will be converted.
    @return the `cv_img` converted to a Qt-compatible PixmapQt-compatible Pixmap.
    """
    try:
        height, width, channel = cv_img.shape
    except Exception as e:
        height, width = cv_img.shape
        channel = 1

    bytesPerLine = channel * width
    q_img = QtGui.QImage(cv_img.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
    return QtGui.QPixmap.fromImage(q_img)

class RobotUI(QtWidgets.QMainWindow):
    def __init__(self):
        super(RobotUI, self).__init__()
        loadUi(UI_PATH, self)

        self.begin_button.clicked.connect(self.SLOT_begin_button)
        self.control_state.clicked.connect(self.SLOT_control_state)
        self.brain_state.clicked.connect(self.SLOT_brain_state)
        self.reset_model.clicked.connect(self.SLOT_reset_model)
        self.navigation_toggle_button.clicked.connect(self.SLOT_toggle_navigation)
        self.vision_button.clicked.connect(self.SLOT_vision_button)

        # Handlers that wrap nodes running in subprocesses
        self._control_node: NodeThread = NodeThread(ROBOT_CONTROL_NODE_NAME)
        self._brain_node: NodeThread = NodeThread(ROBOT_BRAIN_NODE_NAME)
        self._master_node: NodeThread = NodeThread(MASTER_CONTROLLER_NAME)
        self._navigation_node: NodeThread = NodeThread(NAVIGATION_CONTROLLER_NAME)
                
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

        self._read_clueboard_client = actionlib.SimpleActionClient('read_clueboard', ReadClueboardAction)

        # Publisher to publish commands to the robot command topic (robot_control reads this topic, and updates cmd_vel with it continuously)
        self._command_publisher = rospy.Publisher(ROBOT_COMMAND_TOPIC, Twist, queue_size=1)
        self._going_forward = False

        # OpenCV stuff
        self._cv_bridge = CvBridge()
        self._vision_subscriber = rospy.Subscriber(VISION_TOPIC, Image, self._read_vision)
        self._camera_feed_subscriber = rospy.Subscriber(CAMERA_FEED_TOPIC, Image, self._update_camera_feed)
        self._extracted_image_subscriber = rospy.Subscriber(EXTRACTED_IMAGE_TOPIC, Image, self._update_extracted_image)
        self._letters_subscriber = rospy.Subscriber(LETTERS_PUBLISH_TOPIC, Image, self._update_letters)

        self._num_good_points = 0
        self._num_good_points_subscriber = rospy.Subscriber(NUM_GOOD_POINTS_TOPIC, Int16, self._update_good_points)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._update_elapsed_time)
        self.timer.start(TIME_ELAPSED_UPDATE_PERIOD)  # Update every second
        self._time_since_update = 0

    def SLOT_vision_button(self):
        rospy.loginfo("Sending read clueboard...")
        self._send_read_clueboard()
                
    def _update_elapsed_time(self):
        self._time_since_update += TIME_ELAPSED_UPDATE_PERIOD
        self.update_label.setText(f"Time Elapsed since Update: {self._time_since_update}ms") 

    def _send_read_clueboard(self):
        self._read_clueboard_client.wait_for_server()

        goal = ReadClueboardGoal()
        self._read_clueboard_client.send_goal(goal, feedback_cb=self._read_clueboard_callback, done_cb=self._read_clueboard_done_cb)

    def _read_clueboard_callback(self, feedback: ReadClueboardFeedback):
        rospy.loginfo(feedback.clueboard_lock_success)

    def _read_clueboard_done_cb(self, state, result: ReadClueboardResult):
        print(type(state))
        rospy.loginfo(result.clueboard_text)

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

    def _update_letters(self, msg):
        try:
            cv_image: np.ndarray = self._cv_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            cv2.imwrite("./image.png", cv_image)
            threshold_image = convert_cv_to_pixmap(cv_image).scaled(self.threshold_image.size(), aspectRatioMode=True)
            self.threshold_image.setPixmap(threshold_image)

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
            
            # Display the sign
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
        self._brain_node.start()
        self.brain_state.setText("Kill Brain")

    def _enable_nav(self):
        self._master_node.start()
        self._navigation_node.start()
        self.navigation_toggle_button.setText("Kill Nav")

    def _enable_control(self):
        self._control_node.start()
        self.control_state.setText("Kill Control")

    def _kill_control(self):
        self._control_node.kill()
        self.control_state.setText("Enable Control")
    
    def _kill_nav(self):
        self._master_node.kill()
        self._navigation_node.kill()
        self.navigation_toggle_button.setText("Enable Nav")

    def _kill_brain(self):
        self._brain_node.kill()
        self.brain_state.setText("Enable Brain")

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

    def SLOT_toggle_navigation(self):
        rospy.loginfo("Switching navigation state...")

        if self.navigation_toggle_button.text() == "Enable Nav":
            self._enable_nav()
            rospy.loginfo("Enabled navigation!")

        else:
            self._kill_nav()
            rospy.loginfo("Killed navigation!")

    def SLOT_brain_state(self):
        rospy.loginfo("Switching brain state...")
        
        if self.brain_state.text() == "Enable Brain":
            self._enable_brain()
            rospy.loginfo("Enabled control!")

        else:
            self._kill_brain()
            rospy.loginfo("Killed control!")

    def SLOT_reset_model(self, arg = None):
        rospy.loginfo("Trying to Reset Model...")

        self._unload_controllers()  # Stop the controllers to try to avoid weird PID stuff

        self._reset_robot_location()

        self._load_controllers()  # Restart controllers after a brief pause to try to avoid weird PID stuff

        rospy.loginfo("Model Reset!")
        
        return GoForwardResponse()

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

    # def shutdown():
    #     exit()

    # rospy.on_shutdown(lambda: shutdown())  # Connect a callback that gets run when this node gets called to shutdown (just a bit of error logging currently)

    sift_demo_app.show()
    sys.exit(app.exec_())
