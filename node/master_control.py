#!/usr/bin/env python3

import rospy
import threading
from std_msgs.msg import Float64MultiArray, Bool, Float32, String
import time
from robot_controller.msg import ReadClueboardAction, ReadClueboardGoal, ReadClueboardFeedback, ReadClueboardResult
import actionlib
from geometry_msgs.msg import Twist

ROBOT_VELOCITY_TOPIC: str = "/robot_state_command"

# goals dictionary
goals = {
    "sign_start": [0.25, -0.8, 0, -1.20],
    "sign_top_middle": [-1.15, -4.319, 0, 3.555],
    "sign_top_left": [-0.25, -3.7, 0, -1.45], 
    "sign_top_right": [-4.95, -3.50, 0, 1.6],
    "sign_bottom_middle": [-4.9, -0.20, 0, -1.23],
    "sign_bottom_right": [-8.83, -0.834, 0, 3.06],
    "sign_tunnel": [-9.83, -4.7, 0, -0.1],
    "sign_mountain": [-6.4427, -3.69, 0, -0.0803],
    "tunnel_dive": [-8.673, -4.735, 0, 0]
}

SCORE_TRACKER_TOPIC = "/score_tracker"


class MasterController:
    def __init__(self):
        # Publisher for the current goal
        self.current_goal_pub = rospy.Publisher('/current_goal', Float64MultiArray, queue_size=10)
        self.altitute_pub = rospy.Publisher('/robot_desired_altitude', Float32, queue_size=10)
        self._score_tracker_publisher = rospy.Publisher(SCORE_TRACKER_TOPIC, String, queue_size=10)
        self.cmd_vel_pub = rospy.Publisher(ROBOT_VELOCITY_TOPIC, Twist, queue_size=1)


        self._read_clueboard_client = actionlib.SimpleActionClient('read_clueboard', ReadClueboardAction)

        # Subscriber to listen for goal completion
        rospy.Subscriber('/reached_goal', Bool, self.reached_goal_callback)

        self.current_cmd = Twist() 

        # Initialize variables
        self.current_goal = None
        self.goal_reached = False  # Flag to track when a goal is reached
        self._reading_clueboard = False

        # altitudes
        self.navigation_alt = 0.45
        self.reading_alt = 0.2
        self.mountain_alt = 2.25
        self.reading_alt_tunnel = 0.25

        # Thread for publishing the current goal
        self._stop_event = threading.Event()
        self._publisher_thread = threading.Thread(target=self._publish_current_goal)
        self._publisher_thread.start()

        rospy.loginfo("MasterController initialized.")
        self.main_loop()

    def _publish_current_goal(self):
        """Continuously publish the current goal at a set rate."""
        rate = rospy.Rate(1)  # 0.5 Hz
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
        if not self._reading_clueboard:
            self.goal_reached = msg.data
            if self.goal_reached:
                rospy.loginfo("Goal reached. Ready to proceed to the next action.")

    def adjustment_angular(self):
        self.current_goal = None
        self.current_cmd.linear.x = 0.0
        self.current_cmd.linear.y = 0.0
        self.current_cmd.linear.z = 0.0
        self.current_cmd.angular.z = 0.5
        for _ in range(5):
            self.cmd_vel_pub.publish(self.current_cmd)
            rospy.sleep(0.05)
        rospy.sleep(0.5)

    def adjustment_left_yaw(self):
        self.current_goal = None
        self.current_cmd.linear.x = 0.5
        self.current_cmd.linear.y = 0.0
        self.current_cmd.linear.z = 0.0
        self.current_cmd.angular.z = -0.5
        for _ in range(5):
            self.cmd_vel_pub.publish(self.current_cmd)
            rospy.sleep(0.05)
        rospy.sleep(0.5)


    def _send_read_clueboard(self):
        """
        Call this function to tell robot brain to try to read the clueboard.
        """
        self._reading_clueboard = True
        self._read_clueboard_client.wait_for_server()

        goal = ReadClueboardGoal()
        self._read_clueboard_client.send_goal(goal, feedback_cb=self._read_clueboard_callback)  # add arg done_cb=self._read_clueboard_done_cb to attach the done callback

    def _read_clueboard_callback(self, feedback: ReadClueboardFeedback):
        """
        Called when robot brain has (successfully or failed to) acquired a lock on the clueboard
        """
        rospy.loginfo(feedback.clueboard_lock_success)
        print("DONE READING BOARD")
        self._reading_clueboard = False

        # We probably just want to move on if we can't read the clueboard, in which case we should set it to False either way
        if feedback.clueboard_lock_success is True:
            pass
        else:
            # Some repositioning sequence??
            pass

    def _read_clueboard_done_cb(self, state, result: ReadClueboardResult):
        rospy.loginfo(result.clueboard_text)

    def navigate_to_sign(self, sign_name, altitude):
        self.altitute_pub.publish(altitude)
        rospy.sleep(0.5)
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

    def read_sign(self, altitude):
        self.altitute_pub.publish(altitude)
        rospy.sleep(0.5)
        print("READING BOARD")
        self._send_read_clueboard()
        while self._reading_clueboard and not rospy.is_shutdown():
            rospy.sleep(0.5)

    def shutdown(self):
        """Shutdown the publishing thread."""
        self._stop_event.set()
        self._publisher_thread.join()

    def main_loop(self):
        """Main loop to handle sequential actions."""
        rospy.loginfo("Starting action sequence...")

        self.navigate_to_sign("sign_start", self.navigation_alt)
        self.read_sign(self.reading_alt)

        self.navigate_to_sign("sign_top_left", self.navigation_alt)
        self.read_sign(self.reading_alt)

        self.navigate_to_sign("sign_top_middle", self.navigation_alt)
        self.read_sign(self.reading_alt)

        self.navigate_to_sign("sign_top_right", self.navigation_alt)
        self.read_sign(self.reading_alt)

        self.navigate_to_sign("sign_bottom_middle", self.navigation_alt)
        self.read_sign(self.reading_alt)
        self.altitute_pub.publish(self.navigation_alt)
        self.adjustment_angular()

        self.navigate_to_sign("sign_bottom_right", self.navigation_alt)
        self.read_sign(self.reading_alt)
        self.altitute_pub.publish(self.navigation_alt)
        self.adjustment_left_yaw()

        self.navigate_to_sign("sign_tunnel", self.navigation_alt)
        self.read_sign(self.reading_alt_tunnel)   

        self.navigate_to_sign("sign_mountain", self.mountain_alt)
        self.read_sign(self.mountain_alt)
        
        self.navigate_to_sign("tunnel_dive", self.mountain_alt)

        self.altitute_pub.publish(0)
        rospy.sleep(1)
        self._score_tracker_publisher.publish(String("Team4,password,-1,NA"))
        
        rospy.loginfo("Action sequence complete!")



if __name__ == '__main__':
    rospy.init_node('master_controller', anonymous=True)

    time.sleep(3)

    try:
        controller = MasterController()
    except rospy.ROSInterruptException:
        pass
