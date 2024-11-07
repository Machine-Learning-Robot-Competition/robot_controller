#! /usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty, Int8
import time

ROBOT_COMMAND_TOPIC = "/cmd_vel"
ROBOT_TAKEOFF_COMMAND = "/ardrone/takeoff"


class Robot:
    def __init__(self):
        self._cmd_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self._takeoff_publisher = rospy.Publisher('/ardrone/takeoff', Empty, queue_size=1)
        
        self._move = Twist()

        self._begin()

    def _begin(self):
        self._takeoff_publisher.publish(Empty())

    def move(self):
        self._move.linear.z = 1.0

        self._cmd_publisher.publish(self._move)

        rospy.loginfo("Here!")

        time.sleep(3)

        self._move.linear.z = 0.0

        self._cmd_publisher.publish(self._move)



rospy.init_node('drone_brain')
brain = Robot()

time.sleep(10)

rospy.loginfo("Starting!!")

brain.move()

rospy.spin()