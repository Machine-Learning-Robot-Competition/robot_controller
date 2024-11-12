#! /usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist, Vector3
from std_msgs.msg import Empty
from hector_uav_msgs.srv import EnableMotors
from robot_controller.srv import GoForward, GoForwardResponse
import time
import threading


ROBOT_COMMAND_TOPIC = "/cmd_vel"
ROBOT_TAKEOFF_COMMAND = "/ardrone/takeoff"
ENABLE_MOTORS_SERVICE = "/enable_motors"
PUBLISH_RATE = 30  # Rate to publish the drive command to the drone, in Hz
GO_FORWARD_SERVICE = "/go_forward"
ZERO_VECTOR = Vector3(0.0, 0.0, 0.0)  # Just a helper definition 


class Robot:
    def __init__(self):
        rospy.on_shutdown(self._shutdown_hook)  # Connect a callback that gets run when this node gets called to shutdown (just a bit of error logging currently)

        rospy.loginfo("Robot Brain is initializing!")

        # Topic Registrations
        self._cmd_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self._takeoff_publisher = rospy.Publisher('/ardrone/takeoff', Empty, queue_size=1)

        # We are advertising this service
        self._go_forward_service = rospy.Service(GO_FORWARD_SERVICE, GoForward, self._handle_go_forward)
        
        # Movement
        self._move = Twist()
        
        # Publish Threading Stuff
        self._publish_rate = PUBLISH_RATE
        self._publish_thread = None

        self._begin()  # Actual initializtion logic is here

        rospy.loginfo("Robot Brain initialized!")

    def takeoff(self):
        """
        Perform a simple takeoff procedure: rise with v=1.0 for 4.0 seconds, then stop.
        """
        rospy.loginfo("Beginning takeoff procedure! Rising...")

        self.set_action(linear=Vector3(0.0, 0.0, 0.5))

        time.sleep(0.75)

        self.stop()

        rospy.loginfo("Takeoff completed! Stopping...")

    def set_action(self, twist: Twist = None, linear: Vector3 = None, angular: Vector3 = None):
        """
        Set the action state of the drone. 

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

        self._move = twist

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
            self._cmd_publisher.publish(self._move)
            
            rate.sleep()
    
    def _go_forward(self):
        """
        Perform a quick "go forward" operation
        """
        rospy.loginfo("Beginning go forward...")

        self.set_action(linear=Vector3(1.0, 0.0, 0.0))

        time.sleep(1) # 1.0 s

        self.stop()

        rospy.loginfo("Completed go forward!")

    def _handle_go_forward(self, request):
        response = GoForwardResponse()

        try:
            self._go_forward()

            response.success = True

        except Exception:
            response.success = False

        return response


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
    rospy.init_node('robot_brain')

    time.sleep(5)   # Wait a few seconds for the ROS master node to register this node before we start doing anything

    brain = Robot()

    brain.takeoff()

    # This blocks until the node gets a shutdown signal, so that it continues to run 
    # endlessly. Otherwise, it would just exist here and the drone would go brain-dead.
    rospy.spin()
