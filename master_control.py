#!/usr/bin/env python3

import rospy
import threading
from std_msgs.msg import Float64MultiArray, Bool

# Hardcoded goals dictionary
goals = {
    "sign_top_middle": [0, 0, 0, 3.1415],
    "sign_top_left": [0.5, 0.5, 0, 4.4]
}

class MasterController:
    def __init__(self):
        rospy.init_node('master_controller', anonymous=True)

        # Publisher for the current goal
        self.current_goal_pub = rospy.Publisher('/current_goal', Float64MultiArray, queue_size=10)

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

    def main_loop(self):
        """Main loop to handle sequential actions."""
        rospy.loginfo("Starting action sequence...")

        # Go to first goal
        rospy.loginfo("Setting goal to sign_top_left")
        self.current_goal = goals["sign_top_left"]
        self.wait_for_goal()
        rospy.loginfo("chilling!")
        rospy.sleep(10)

        self.current_goal = goals["sign_top_middle"]
        self.wait_for_goal()
        rospy.loginfo("chilling!")
        rospy.sleep(10)

        rospy.loginfo("Action sequence complete!")

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


if __name__ == '__main__':
    try:
        controller = MasterController()
    except rospy.ROSInterruptException:
        pass
