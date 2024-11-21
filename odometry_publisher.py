#!/usr/bin/env python3
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Vector3Stamped, Point, Quaternion
from sensor_msgs.msg import Imu
import tf

class OdometryPublisher:
    def __init__(self):
        rospy.loginfo('OdometryPublisher node initialized!')
        rospy.init_node('odometry_publisher', log_level=rospy.DEBUG)
        
        # Create a publisher for odometry
        self.odom_pub = rospy.Publisher('/odom_custom', Odometry, queue_size=10)

        # Initialize transformation broadcaster
        self.br = tf.TransformBroadcaster()
        
        # Initial pose
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0  # Added z position
        self.theta = 3.1415926535897  # Robot orientation (heading)
        self.last_time = rospy.Time.now()

        # Subscribe to the fix_velocity topic from hector_quadrotor (geometry_msgs/Vector3Stamped)
        self.velocity_sub = rospy.Subscriber('/fix_velocity', Vector3Stamped, self.velocity_callback)
        
        # Subscribe to the IMU topic for angular velocity data
        self.imu_sub = rospy.Subscriber('/raw_imu', Imu, self.imu_callback)

        # Initialize angular velocity (vth)
        self.vth = 0.0

        self.publish_rate = 30



    def velocity_callback(self, msg):
        # rospy.loginfo('Velocity callback triggered')
        current_time = rospy.Time.now()
        dt = (current_time - self.last_time).to_sec()

        # Check if it's time to publish (based on rate)
        if dt < 1.0 / self.publish_rate:
            return  # If it's not time yet, exit the callback

        # Extract linear velocity from Vector3Stamped message
        vx = msg.vector.x
        vy = msg.vector.y
        vz = msg.vector.z  # Vertical velocity

        # Integrating linear velocities to update position
        self.x += vx * dt
        self.y += vy * dt
        self.z += vz * dt  # Integrating vertical position

        # Update theta (orientation) from angular velocity
        self.theta += self.vth * dt  # Integrate angular velocity

        # Create odometry message
        odom = Odometry()
        odom.header.stamp = current_time
        odom.header.frame_id = "odom_custom"
        odom.child_frame_id = "base_footprint_custom"

        # Assign position as geometry_msgs/Point
        odom.pose.pose.position = Point(self.x, self.y, self.z)

        # Assign orientation as geometry_msgs/Quaternion (using updated theta)
        odom.pose.pose.orientation = Quaternion(*tf.transformations.quaternion_from_euler(0, 0, self.theta))

        # Velocity
        odom.twist.twist.linear.x = vx
        odom.twist.twist.linear.y = vy
        odom.twist.twist.linear.z = vz
        odom.twist.twist.angular.z = self.vth

        # Publish the odometry message
        self.odom_pub.publish(odom)
        
        # Broadcast the transform
        rospy.loginfo(f"Broadcasting transform: base_footprint -> odom_custom")
        rospy.loginfo(f"Position: {self.x}, {self.y}, {self.z}")
        rospy.loginfo(f"Rotation: {self.theta}")

        self.br.sendTransform(
            (self.x, self.y, self.z),  # Updated to broadcast z position
            tf.transformations.quaternion_from_euler(0, 0, self.theta),
            current_time,
            "base_footprint_custom",
            "odom_custom"
        )

        self.last_time = current_time


    def imu_callback(self, msg):
        # Extract angular velocity from the IMU message (around the z-axis)
        self.vth = msg.angular_velocity.z  # This is the angular velocity around the z-axis (rotation)
        # rospy.loginfo(f"Angular velocity (z-axis): {self.vth}")

if __name__ == '__main__':
    try:
        odometry_publisher = OdometryPublisher()
        rospy.loginfo("Starting rospy.spin()")
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS Interrupt Exception")
