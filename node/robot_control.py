#! /usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist, Vector3, Vector3Stamped
from std_msgs.msg import String, Empty, Float32
from node_utils import PIDController
from hector_uav_msgs.srv import EnableMotors
from robot_controller.srv import GoForward, GoForwardResponse
import time
import threading


# Topic and Service Names
ROBOT_VELOCITY_TOPIC = "/cmd_vel"
ROBOT_TAKEOFF_COMMAND = "/ardrone/takeoff"
ENABLE_MOTORS_SERVICE = "/enable_motors"
GO_FORWARD_SERVICE = "/go_forward"
ROBOT_COMMAND_TOPIC = "/robot_state_command"
ROBOT_Z_POSITION_TOPIC = "/robot_desired_altitude"
SCORE_TRACKER_TOPIC = "/score_tracker"
VELOCITY_TOPIC = "/fix_velocity"

# Other Declarations
ZERO_VECTOR: Vector3 = Vector3(0.0, 0.0, 0.0)  # Just a helper definition 
PUBLISH_RATE: int = 30  # Rate to publish the drive command to the drone, in Hz

altitude_kp = 1.0
altitude_ki = 0.4
altitude_kd = 0.1


class RobotControlNode:
    """
    Node to control the robot's movement
    """
    def __init__(self):
        rospy.on_shutdown(self._shutdown_hook)  # Connect a callback that gets run when this node gets called to shutdown (just a bit of error logging currently)

        rospy.loginfo("Robot Control is initializing!")

       # Time
        self._current_time = time.time()

        # Topic Registrations
        self._cmd_publisher = rospy.Publisher(ROBOT_VELOCITY_TOPIC, Twist, queue_size=1)
        self._takeoff_publisher = rospy.Publisher(ROBOT_TAKEOFF_COMMAND, Empty, queue_size=1)
        self._command_subscriber = rospy.Subscriber(ROBOT_COMMAND_TOPIC, Twist, callback=self._set_action_callback, queue_size=1)
        self._score_tracker_publisher = rospy.Publisher(SCORE_TRACKER_TOPIC, String, queue_size=10)
        self._desired_altitude_subscriber = rospy.Subscriber(ROBOT_Z_POSITION_TOPIC, Float32, queue_size=1, callback=self._update_desired_altitude)
        self._velocity_update_subscriber = rospy.Subscriber(VELOCITY_TOPIC, Vector3Stamped, queue_size=10, callback=self._update_velocity_callback)
        
        # Movement
        self._move = Twist()
        self._altitude = 0.0
        self._desired_altitude = 0.0
        self._altitude_pid = PIDController(altitude_kp, altitude_ki, altitude_kd)
        
        # Publish Threading Stuff
        self._publish_rate = PUBLISH_RATE
        self._publish_thread = None

        self._begin()  # Actual initializtion logic is here

        rospy.loginfo("Robot Control initialized!")

    def takeoff(self):
        """
        Perform a simple takeoff procedure: rise with v=1.0 for 4.0 seconds, then stop.
        """
        rospy.loginfo("Beginning takeoff procedure! Rising...")

        self._desired_altitude = 0.33

        rospy.loginfo("Takeoff completed!")

    def _update_velocity_callback(self, msg: Vector3Stamped):
        z_velocity = msg.vector.z

        current_time = time.time()
        delta_t = current_time - self._current_time

        self._current_time = current_time
        self._altitude += z_velocity * delta_t

    def _update_desired_altitude(self, desired_altitude: Float32):
        self._desired_altitude = float(desired_altitude.data)

        rospy.loginfo(f"Acquired New Desired Altitude: {self._desired_altitude}")

    def _update_z_velocity(self):
        update_period: float = 1.0 / PUBLISH_RATE # Period in seconds
        z_velocity: float = self._altitude_pid.compute(self._desired_altitude, self._altitude, update_period)

        self.set_z_action(z_velocity)

    def set_z_action(self, z: float):
        self._move.linear.z = z

    def set_action(self, twist: Twist = None, linear: Vector3 = None, angular: Vector3 = None):
        """
        Set the action state of the drone. Does not affect the z-velocity, which is internally controlled.

        If `twist` is set, then it will override `linear` and `angular`. `linear` and `angular` are defaulted to the zero vector if not provided.

        :param Twist twist: a Twist containing the complete action state of the drone, overrides `linear` and `angular`
        :param Vector3 linear: describes the linear motion (x, y, z) of the drone. Unused if `twist` is set. Defaulted to the zero vector.
        :param Vector3 angular: describes the angular motion (r, p, y) of the drone. Unused if `twist` is set. Defaulted to the zero vector.
        """
        if twist is None:
            twist = Twist()

            if linear is None:
                linear = ZERO_VECTOR
            if angular is None:
                angular = ZERO_VECTOR
            
            twist.linear = linear
            twist.angular = angular

        twist.linear.z = self._move.linear.z
        self._move = twist

    def _set_action_callback(self, command: Twist):
        """
        Callback to set the move state
        """
        self.set_action(twist=command)

    def stop(self):
        """
        Stop the drone (try to cease linear and angular motion). No effect if already stopped.
        """
        self._move = Twist(linear=Vector3(0.0, 0.0, 0.0), angular=Vector3(0.0, 0.0, 0.0))

    def _publish_command(self, frequency):
        """
        This is a blocking method. Infinitely loop at `frequency` Hz, publishing the drive command to the drone.

        :param int frequency: the publishing frequency, in Hz
        """        
        rate = rospy.Rate(frequency)

        while not rospy.is_shutdown():
            # Altitude PID Update
            self._update_z_velocity()

            self._cmd_publisher.publish(self._move)
            
            rate.sleep()

    def _begin(self):
        """
        Perform any initialization steps before the robot can be used.
        """
        self._enable_motors_service()  # We need to enable the motors before we can do anything

        self._takeoff_publisher.publish(Empty())    # Tell the robot to take off (just listen for drive commands, really, does not cause the robot to move)

        # Start a thread to run a loop that will publish to the drive state topic with our desired drive state since
        # the drone needs constant updates (even if desired velocity is (0.0, 0.0, 0.0)) or it will fall out of the sky!
        self._publish_thread = threading.Thread(target=self._publish_command, args=(self._publish_rate, ))
        self._publish_thread.start()

    def _enable_motors_service(self):
        """
        According to http://wiki.ros.org/hector_quadrotor/Tutorials/Quadrotor%20indoor%20SLAM%20demo, and experimental verification ( >:() ), we need
        to make a service call to enable the motors before doing anything else.
        """
        # Services are like topics, but more like an HTTP endpoint where you make a request then get a response.
        rospy.wait_for_service(ENABLE_MOTORS_SERVICE)  # Make sure the service is available.
        try:
            enable_motors_service = rospy.ServiceProxy(ENABLE_MOTORS_SERVICE, EnableMotors)  # Get an object to make the service call 
            response = enable_motors_service(True)                                           # Make the call

            rospy.loginfo(f"Enable Motors Service Call: {response}")                         # Log the response

        except rospy.ServiceException as err:
            rospy.logerr(f"Enable Motors Service Call Failed: {err}")                       

    def _shutdown_hook(self):
        """
        Executed upon a shutdown.
        """
        rospy.loginfo("Shutting down robot brain!")


if __name__ == "__main__":
    rospy.init_node('robot_control')

    time.sleep(3)   # Wait a few seconds for the ROS master node to register this node before we start doing anything

    robot_control_node = RobotControlNode()

    robot_control_node.takeoff()

    # This blocks until the node gets a shutdown signal, so that it continues to run 
    # endlessly. Otherwise, it would just exist here and the drone would go brain-dead.
    rospy.spin()
