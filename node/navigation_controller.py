#!/usr/bin/env python3
import rospy
import numpy as np
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from std_msgs.msg import Float64MultiArray, Bool
from tf.transformations import euler_from_quaternion
import time
import threading
from geometry_msgs.msg import Vector3Stamped
from sensor_msgs.msg import Imu


ROBOT_VELOCITY_TOPIC: str = "/robot_state_command"
VELOCITY_PUBLISH_RATE: int = 30  # Hz


class NavigationController:
    def __init__(self):
        self.pub_rate = VELOCITY_PUBLISH_RATE  # Hz

        # publishers
        self.cmd_vel_pub = rospy.Publisher(ROBOT_VELOCITY_TOPIC, Twist, queue_size=1)
        self.reached_goal_pub = rospy.Publisher('/reached_goal', Bool, queue_size=10)

        # subscribers
        rospy.Subscriber('/localization_pose', PoseWithCovarianceStamped, self.localization_callback)
        rospy.Subscriber('/current_goal', Float64MultiArray, self.goal_callback)

        rospy.loginfo("Publisher and subscriber initialized!")

        # Initialize observation variables
        self.relative_pose = None
        self.current_pose = None
        self.last_pose = None
        self.goal_position = None

        self.current_cmd = Twist() 
        self._stop_event = threading.Event()
        self._publisher_thread = threading.Thread(target=self._publish_cmd_vel)
        self._publisher_thread.start()


        # PID controller gains
        self.kp = 1.0
        self.ki = 0.00 # no integral gain for now
        self.kd = 0.011
        self.integral_error = np.zeros(2)
        self.previous_error = np.zeros(2)
        self.dt = 1.0 / self.pub_rate  # Time step based on the publishing rate

        # Params for evaluating if the goal was reached
        self.goal_tolerance = 0.3
        self.reached_goal = False
        self.reached_goal_count = 0
        self.reached_goal_threshold = 3 # number of time needed to be at goal to be considered done (account for oscillations)

    def _publish_cmd_vel(self):
        """Continuously publish the current velocity command to /cmd_vel."""
        rate = rospy.Rate(self.pub_rate)
        while not self._stop_event.is_set() and not rospy.is_shutdown():
            self.cmd_vel_pub.publish(self.current_cmd)
            rate.sleep()


    def localization_callback(self, msg):
        """Callback for /localization_pose topic."""

        # Ensure goal_position is not None
        if self.goal_position is None:
            rospy.logwarn("Goal position is not set. Waiting for /current_goal message.")
            return
        
        pose = msg.pose.pose
        orientation = pose.orientation
        theta = self.get_theta_from_orientation(orientation)
        self.current_pose = np.array([pose.position.x, pose.position.y, pose.position.z, theta])
        self.relative_pose = self.goal_position[:2] - self.current_pose[:2]
        print(f'relative pose: {self.relative_pose}')
        print(f'current pose: {self.current_pose}')

        self.update_pid_controller()
    
    def goal_callback(self, msg):
        """Callback to set goal_position if the received goal is different."""
        # Extract the goal position from the message
        new_goal_position = msg.data  # Assuming msg is a Float64MultiArray
        
        # Check if the new goal is different from the current goal
        if self.goal_position is None or new_goal_position != self.goal_position:
            rospy.loginfo(f"Updating goal position to: {new_goal_position}")
            self.goal_position = new_goal_position
            self.reached_goal = False
            self.reached_goal_count = 0
        else:
            rospy.loginfo("Received goal position is the same as the current goal. No update.")

    
    def update_pid_controller(self):
        """Calculate and publish a velocity vector using a PID controller."""
        if self.goal_position is None:
            self.current_cmd.linear.x = 0.0
            self.current_cmd.linear.y = 0.0
            self.current_cmd.linear.z = 0.0
            self.current_cmd.angular.z = 0.0
            return

        error = self.relative_pose[:2]
        error_magnitude = np.linalg.norm(error)
        print(f'DISTANCE {error_magnitude}')

        # Check if the goal has been reached
        if error_magnitude < self.goal_tolerance:
            self.reached_goal_count += 1
            if self.reached_goal_count >= self.reached_goal_threshold:
                if not self.reached_goal:
                    self.reached_goal = True
                    rospy.loginfo("Goal reached!")
                    self.reached_goal_pub.publish(Bool(data=True))
            else:
                self.reached_goal_pub.publish(Bool(data=False))
        else:
            self.reached_goal_count = 0
            self.reached_goal_pub.publish(Bool(data=False))

        # Stop linear motion if the goal is reached, align with desired heading
        if self.reached_goal:
            alignment_goal = self.goal_position[3]  # Desired heading
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
        else:
            # Regular PID-based navigation
            desired_heading = np.arctan2(error[1], error[0])  # Angle toward the goal
            current_heading = self.current_pose[3]

            angular_error = desired_heading - current_heading
            angular_error = np.arctan2(np.sin(angular_error), np.cos(angular_error))

            rotation_matrix = np.array([
                [np.cos(current_heading), np.sin(current_heading)],
                [-np.sin(current_heading), np.cos(current_heading)]
            ])
            error_in_robot_frame = rotation_matrix @ error

            self.integral_error += error_in_robot_frame * self.dt
            derivative_error = (error_in_robot_frame - self.previous_error) / self.dt

            control_output = (
                self.kp * error_in_robot_frame +
                self.ki * self.integral_error +
                self.kd * derivative_error
            )

            scaling_factor = 1
            if error_magnitude < 1.4:
                scaling_factor = 0.3 * error_magnitude

            
            if np.linalg.norm(control_output) > .8:
                control_output =  control_output / np.linalg.norm(control_output) * 0.8

            control_output *= scaling_factor

            self.current_cmd.linear.x = control_output[0]
            self.current_cmd.linear.y = 0.0
            self.current_cmd.linear.z = 0.0
            self.current_cmd.angular.z = 2.0 * angular_error

            self.previous_error = error_in_robot_frame

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
    rospy.init_node('navigation_controller', anonymous=True)

    time.sleep(3)

    try:
        navigation_controller = NavigationController()
        rospy.loginfo("Starting rospy.spin()")
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS Interrupt Exception")