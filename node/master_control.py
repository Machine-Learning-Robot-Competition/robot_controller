#!/usr/bin/env python3

import rospy
import threading
from std_msgs.msg import Float64MultiArray, Bool, Float32
import time


# Hardcoded goals dictionary
goals = {
    "sign_start": [0.033, -0.8, 0, -1.20],
    "sign_top_middle": [-1.38, -4.019, 0, 3.805],
    "sign_top_left": [-0.094, -3.8, 0, -2.4], 
    "sign_top_right": [-5.15, -3.40, 0, 1.07],
    "sign_bottom_middle": [-4.74, -0.365, 0, -1.23],
    "sign_bottom_right": [-8.90, -0.834, 0, 3.06],
    "sign_tunnel": [-9.93, -4.7, 0, 0.02]
}
navigation_alt = 0.3
reading_alt = 0.2
mountain_alt = 2

class MasterController:
    def __init__(self):
        # Publisher for the current goal
        self.current_goal_pub = rospy.Publisher('/current_goal', Float64MultiArray, queue_size=10)
        self.altitute_pub = rospy.Publisher('/robot_desired_altitude', Float32, queue_size=10)

        # Subscriber to listen for goal completion
        rospy.Subscriber('/reached_goal', Bool, self.reached_goal_callback)

        # Initialize variables
        self.current_goal = None
        self.goal_reached = False  # Flag to track when a goal is reached

        # Thread for publishing the current goal
        self._stop_event = threading.Event()
        self._publisher_thread = threading.Thread(target=self._publish_current_goal)
        self._publisher_thread.start()

        rospy.loginfo("MasterController initialized.")
        self.main_loop()

    def _publish_current_goal(self):
        """Continuously publish the current goal at a set rate."""
        rate = rospy.Rate(0.5)  # 0.5 Hz
        while not self._stop_event.is_set() and not rospy.is_shutdown():
            if self.current_goal is not None:
                # Prepare the Float64MultiArray message
                current_goal_msg = Float64MultiArray()
                current_goal_msg.data = self.current_goal
                self.current_goal_pub.publish(current_goal_msg)
                # rospy.loginfo(f"Published current goal: {self.current_goal}")
            rate.sleep()

    def reached_goal_callback(self, msg):
        """Callback to update the goal_reached flag."""
        self.goal_reached = msg.data
        if self.goal_reached:
            rospy.loginfo("Goal reached. Ready to proceed to the next action.")

    def navigate_to_sign(self, sign_name):
        self.altitute_pub.publish(navigation_alt)

        # Go to first goal
        rospy.loginfo(f'Setting goal to {sign_name}')
        self.current_goal = goals[sign_name]
        self.wait_for_goal()

    def wait_for_goal(self):
        """Wait until the current goal is reached."""
        rospy.loginfo("Waiting for goal to be reached...")
        while not self.goal_reached and not rospy.is_shutdown():
            rospy.sleep(0.5)
        self.goal_reached = False  # Reset for the next goal

    def shutdown(self):
        """Shutdown the publishing thread."""
        self._stop_event.set()
        self._publisher_thread.join()

    def main_loop(self):
        """Main loop to handle sequential actions."""
        rospy.loginfo("Starting action sequence...")

        self.navigate_to_sign("sign_start")
        rospy.loginfo("chilling!")
        self.altitute_pub.publish(reading_alt)
        rospy.sleep(5)

        self.navigate_to_sign("sign_top_left")
        rospy.loginfo("chilling!")
        self.altitute_pub.publish(reading_alt)
        rospy.sleep(5)


        self.navigate_to_sign("sign_top_middle")
        rospy.loginfo("chilling!")
        self.altitute_pub.publish(reading_alt)
        rospy.sleep(5)


        self.navigate_to_sign("sign_top_right")
        rospy.loginfo("chilling!")
        self.altitute_pub.publish(reading_alt)
        rospy.sleep(5)


        self.navigate_to_sign("sign_bottom_middle")
        rospy.loginfo("chilling!")
        self.altitute_pub.publish(reading_alt)
        rospy.sleep(5)


        self.navigate_to_sign("sign_bottom_right")
        rospy.loginfo("chilling!")
        self.altitute_pub.publish(reading_alt)
        rospy.sleep(5)

        self.navigate_to_sign("sign_tunnel")
        rospy.loginfo("chilling!")
        self.altitute_pub.publish(reading_alt)
        rospy.sleep(5)
        
        self.altitute_pub.publish(navigation_alt)

        rospy.loginfo("Action sequence complete!")



if __name__ == '__main__':
    rospy.init_node('master_controller', anonymous=True)

    time.sleep(3)

    try:
        controller = MasterController()
    except rospy.ROSInterruptException:
        pass
