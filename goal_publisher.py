#!/usr/bin/env python3

import rospy
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import Float64MultiArray, MultiArrayDimension

class GoalPublisher:
    def __init__(self):
        rospy.init_node('goal_publisher', anonymous=True)

        self.labeled_points = {}  # Store labeled points

        # Publisher for goal positions
        self.goal_positions_pub = rospy.Publisher('/goal_positions', Float64MultiArray, queue_size=10)

        # Subscribe to /labels to get the location of labeled points
        rospy.Subscriber('/labels', MarkerArray, self.labels_callback)

        rospy.loginfo("GoalPublisher initialized and waiting for data.")
        rospy.spin()

    def labels_callback(self, msg):
        """Process labeled points and publish them as an array of arrays."""
        goal_positions = Float64MultiArray()  # Initialize the message

        # Prepare the list of goal positions
        data = []
        num_goals = 0

        rospy.loginfo("\nReceived Labeled Goals:\n")
        for marker in msg.markers:
            label = marker.text
            # Include only valid labels
            if not label[0].isdigit() or label[0] == "m":
                position = marker.pose.position
                self.labeled_points[label] = {
                    'x': position.x,
                    'y': position.y,
                    'z': position.z
                }
                # Append [x, y, z] as a single goal
                data.extend([position.x, position.y, position.z])
                num_goals += 1

                # Print the formatted goal information
                rospy.loginfo(f"  Goal Name: {label}")
                rospy.loginfo(f"    Coordinates: x = {position.x:.2f}, y = {position.y:.2f}, z = {position.z:.2f}")

        # Populate the layout dimensions for a 2D array
        if num_goals > 0:
            goal_positions.layout.dim.append(MultiArrayDimension(label="goals", size=num_goals, stride=num_goals * 3))
            goal_positions.layout.dim.append(MultiArrayDimension(label="coordinates", size=3, stride=3))
            goal_positions.layout.data_offset = 0

            # Assign the flat data array
            goal_positions.data = data

            # Publish the goal positions
            self.goal_positions_pub.publish(goal_positions)
            rospy.loginfo("\nPublished Goal Positions:\n")
            for label, coords in self.labeled_points.items():
                rospy.loginfo(f"  {label}: [x = {coords['x']:.2f}, y = {coords['y']:.2f}, z = {coords['z']:.2f}]")
            rospy.loginfo("\n")

if __name__ == '__main__':
    try:
        GoalPublisher()
    except rospy.ROSInterruptException:
        pass
