#!/usr/bin/env python3

import rospy
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import PoseWithCovarianceStamped
from math import sqrt

class DistanceToPOIs:
    def __init__(self):
        rospy.init_node('distance_to_pois', anonymous=True)

        self.labeled_points = {}
        self.robot_position = None

        # Subscribe to /labels to get the location of labeled points
        rospy.Subscriber('/labels', MarkerArray, self.labels_callback)

        # Subscribe to /localization_pose to get the robot's current position
        rospy.Subscriber('/localization_pose', PoseWithCovarianceStamped, self.localization_callback)

        # Start the main loop
        self.main_loop()

    def labels_callback(self, msg):
        # Iterate through each marker in the MarkerArray
        for marker in msg.markers:
            label = marker.text
            if not label[0].isdigit():
                position = marker.pose.position
                self.labeled_points[label] = {
                    'x': position.x,
                    'y': position.y,
                    'z': position.z
                }
                # rospy.lo/info(f"Stored waypoint '{label}' at position ({position.x}, {position.y}, {position.z})")

    def localization_callback(self, msg):
        # Update the robot's current position from /localization_pose
        self.robot_position = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        )
        # rospy.loginfo(f"Updated robot position: {self.robot_position}")  # Debugging output

    def calculate_distance(self, robot_position, poi_position):
        # Calculate the Euclidean distance between two points
        x_robot, y_robot, z_robot = robot_position
        x_poi, y_poi, z_poi = poi_position['x'], poi_position['y'], poi_position['z']
        distance = sqrt((x_poi - x_robot) ** 2 + (y_poi - y_robot) ** 2 + (z_poi - z_robot) ** 2)
        return distance

    def main_loop(self):
        rate = rospy.Rate(1)  # 1 Hz update rate
        while not rospy.is_shutdown():
            # Check if the robot's position is available
            if self.robot_position:
                for poi_name, poi_position in self.labeled_points.items():
                    distance = self.calculate_distance(self.robot_position, poi_position)
                    rospy.loginfo(f"Distance to '{poi_name}': {distance:.2f} meters")
            else:
                rospy.logwarn("Robot position not available, skipping distance calculation.")

            rate.sleep()

if __name__ == '__main__':
    try:
        DistanceToPOIs()
    except rospy.ROSInterruptException:
        pass
