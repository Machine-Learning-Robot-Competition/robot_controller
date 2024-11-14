#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist

class SkidSteerController:
    def __init__(self):
        rospy.init_node('skid_steer_controller', anonymous=True)
        
        # Subscriber to localization pose
        self.pose_subscriber = rospy.Subscriber('/rtabmap/localization_pose', PoseWithCovarianceStamped, self.pose_callback)
        
        # Set up the publisher for movement commands
        self.cmd_vel_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        
        self.current_pose = None
        
        self.linear_speed = 0.5  # m/s
        self.angular_speed = 0.3  # rad/s

    def pose_callback(self, msg):
        """
        Callback function that gets executed each time a new pose message is received.
        """
        self.current_pose = msg.pose.pose
        rospy.loginfo(f"Current position: x={self.current_pose.position.x}, y={self.current_pose.position.y}")

    def move_robot(self):
        """
        Function to publish movement commands based on the current pose.
        """
        rate = rospy.Rate(10)  # Loop at 10 Hz
        move_cmd = Twist()
        
        while not rospy.is_shutdown():
            if self.current_pose is None:
                rospy.logwarn("Waiting for pose data...")
                rate.sleep()
                continue
            
            # Define a simple movement strategy, e.g., move forward constantly
            move_cmd.linear.x = self.linear_speed  # Constant forward movement
            move_cmd.angular.z = self.angular_speed  # Constant turn
            
            # Publish the command
            self.cmd_vel_publisher.publish(move_cmd)
            
            # Sleep to maintain the loop rate
            rate.sleep()

if __name__ == '__main__':
    try:
        controller = SkidSteerController()
        controller.move_robot()
    except rospy.ROSInterruptException:
        pass
