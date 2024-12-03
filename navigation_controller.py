#!/usr/bin/env python3
import rospy
import numpy as np
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Vector3Stamped, Point, Quaternion, Twist, PoseWithCovarianceStamped
from tf.transformations import quaternion_from_euler, euler_from_quaternion

import threading

class NavigationController:
    def __init__(self):
        self.goal_position = np.array([2, -1.0, 0.0, 1.57])

        rospy.init_node('drone_sim_env', anonymous=True)
        rospy.loginfo("ROS node initialized successfully!")

        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        rospy.Subscriber('/localization_pose', PoseWithCovarianceStamped, self.localization_callback)
        rospy.loginfo("Publisher and subscriber initialized!")

        # Initialize observation variables
        self.relative_pose = None
        self.current_pose = None
        self.last_pose = None

        self.current_cmd = Twist() 
        self._stop_event = threading.Event()
        self._publisher_thread = threading.Thread(target=self._publish_cmd_vel)
        self._publisher_thread.start()

        # PID controller gains
        self.kp = 1.0
        self.ki = 0.00 # no integral gain for now
        self.kd = 0.0050
        self.integral_error = np.zeros(2)
        self.previous_error = np.zeros(2)
        self.dt = 1.0 / 45  # Time step based on the publishing rate

        self.goal_tolerance = 0.25
        self.reached_goal = False
        self.reached_goal_count = 0
        self.reached_goal_threshold = 3 # number of time needed to be at goal to be considered done

    def _publish_cmd_vel(self):
        """Continuously publish the current velocity command to /cmd_vel."""
        rate = rospy.Rate(45) # publishing rate in Hz
        while not self._stop_event.is_set() and not rospy.is_shutdown():
            self.cmd_vel_pub.publish(self.current_cmd)
            rate.sleep()

    def localization_callback(self, msg):
        """Callback for /localization_pose topic."""
        pose = msg.pose.pose
        orientation = pose.orientation
        theta = self.get_theta_from_orientation(orientation)
        self.current_pose = np.array([pose.position.x, pose.position.y, pose.position.z, theta])
        self.relative_pose = self.goal_position[:2] - self.current_pose[:2]
        print(f'relative pose: {self.relative_pose}')
        print(f'current pose: {self.current_pose}')
        self.update_pid_controller()
    
    def update_pid_controller(self):
        """Calculate and publish a velocity vector using a PID controller."""
        # Error in the global frame (x, y)
        error = self.relative_pose[:2]
        
        # Desired heading angle in global frame
        desired_heading = np.arctan2(error[1], error[0])  # Angle toward the goal

        # Robot's current heading (yaw)
        current_heading = self.current_pose[3]

        # Angular error (normalize to [-pi, pi])
        angular_error = desired_heading - current_heading
        angular_error = np.arctan2(np.sin(angular_error), np.cos(angular_error))

        # Rotate error into the robot's frame of reference
        rotation_matrix = np.array([
            [np.cos(current_heading), np.sin(current_heading)],
            [-np.sin(current_heading), np.cos(current_heading)]
        ])
        error_in_robot_frame = rotation_matrix @ error

        # PID calculations for linear velocity in the robot's frame
        self.integral_error += error_in_robot_frame * self.dt
        derivative_error = (error_in_robot_frame - self.previous_error) / self.dt

        control_output = (
            self.kp * error_in_robot_frame +
            self.ki * self.integral_error +
            self.kd * derivative_error
        )

        # Normalize the control vector to have a magnitude of 1
        if np.linalg.norm(control_output) > 1:
            control_output = control_output / np.linalg.norm(control_output)

        # Angular velocity control (proportional term only)
        angular_velocity = 2.0 * angular_error  # Gain of 2.0 for angular control

        # Update the Twist message
        self.current_cmd.linear.x = control_output[0]  # Forward velocity
        self.current_cmd.linear.y = 0.0  # Avoid lateral movement
        self.current_cmd.linear.z = 0.0
        self.current_cmd.angular.z = angular_velocity

        # Save the error for the next derivative calculation
        self.previous_error = error_in_robot_frame

        if np.linalg.norm(error) < self.goal_tolerance:
            self.reached_goal_count += 1
            if self.reached_goal_count >= self.reached_goal_threshold:
                self.reached_goal = True
                self.reached_goal_count = 0
            
        # if we reached the goal then just align with the desired heading
        if self.reached_goal:
            # If within the goal threshold, stop linear motion
            rospy.loginfo("Reached goal! Aligning!")
            alignment_goal = self.goal_position[3]
            self.current_cmd.linear.x = 0.0
            self.current_cmd.linear.y = 0.0
            self.current_cmd.linear.z = 0.0

            # Align to the desired heading
            current_heading = self.current_pose[3]
            angular_error = alignment_goal - current_heading
            angular_error = np.arctan2(np.sin(angular_error), np.cos(angular_error))  # Normalize to [-pi, pi]

            # Angular velocity control (proportional term only)
            angular_velocity = 2.0 * angular_error  # Gain of 2.0 for angular control
            self.current_cmd.angular.z = angular_velocity

            # If aligned within a small angular tolerance, stop rotation
            angular_tolerance = 0.05  # 0.05 radians (~3 degrees)
            if abs(angular_error) < angular_tolerance:
                self.current_cmd.angular.z = 0.0



    def get_theta_from_orientation(self, orientation):
        """
        Extract the yaw angle (theta) from a quaternion.

        Args:
            orientation: A geometry_msgs/Quaternion message containing x, y, z, w.

        Returns:
            theta: The yaw angle in radians.
        """
        quaternion = (orientation.x, orientation.y, orientation.z, orientation.w)
        _, _, yaw = euler_from_quaternion(quaternion)
        return yaw



if __name__ == '__main__':
    try:
        navigation_controller = NavigationController()
        rospy.loginfo("Starting rospy.spin()")
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS Interrupt Exception")