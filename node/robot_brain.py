#! /usr/bin/env python3

import threading
import rospy
import time
import pathlib
import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Int16
from cv_bridge import CvBridge


REFERENCE_IMAGE_PATH: str = str(pathlib.Path(__file__).absolute().parent.parent / "media" / "reference_image.png")  # Image Dimensions: (834, 1390)
CAMERA_FEED_TOPIC: str = "/front_cam/camera/image"
RESULT_PUBLISH_TOPIC: str = "/robot_brain/vision"
GOOD_POINTS_TOPIC: str = "/robot_brain/good_points"
EXTRACTED_IMAGE_TOPIC: str = "/robot_brain/extracted_image"


class RobotBrainNode:
    def __init__(self):
        rospy.on_shutdown(self._shutdown_hook)  # Connect a callback that gets run when this node gets called to shutdown (just a bit of error logging currently)

        rospy.loginfo("Initializing brain...")

        self._sift: cv2.SIFT = cv2.SIFT_create()
        index_params = dict(algorithm=0, trees=5)
        search_params = dict()
        self._flann = cv2.FlannBasedMatcher(index_params, search_params)

        self._h, self._w = self._prepare_reference_image()

        self._cv_bridge = CvBridge()

        self._publish_rate = 1  # H[z

        self._vision = None
        self._extracted_image = None

        self._camera_subscriber = rospy.Subscriber(CAMERA_FEED_TOPIC, Image, self.update, queue_size=1)
        self._vision_publisber = rospy.Publisher(RESULT_PUBLISH_TOPIC, Image, queue_size=1)
        self._extracted_image_publisher = rospy.Publisher(EXTRACTED_IMAGE_TOPIC, Image, queue_size=1)
        self._good_points_publisber = rospy.Publisher(GOOD_POINTS_TOPIC, Int16, queue_size=1)

        # Start a thread to run a loop that will publish to the vision topic with what we are seeing
        self._publish_thread = threading.Thread(target=self._publish_command, args=(self._publish_rate, ))
        self._publish_thread.start()

        rospy.loginfo("Brain Initialized!")

    def _publish_command(self, frequency):
        """
        This is a blocking method. Infinitely loop at `frequency` Hz, publishing our vision.

        :param int frequency: the publishing frequency, in Hz
        """        
        rate = rospy.Rate(frequency)

        while not rospy.is_shutdown():
            if self._vision is not None:
                try:
                    vision_msg = self._cv_bridge.cv2_to_imgmsg(self._vision)
                    extracted_image_msg = self._cv_bridge.cv2_to_imgmsg(self._extracted_image)

                    self._vision_publisber.publish(vision_msg)
                    self._extracted_image_publisher.publish(extracted_image_msg)

                except Exception as e:
                    rospy.logerr(e)
            
            rate.sleep()

    
    def _prepare_reference_image(self):
        """
        Load the reference image and process it with SIFT.

        Return the height and width of the loaded reference image.
        """
        template_image = cv2.imread(REFERENCE_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
        self._image_keypoints, self._image_descriptors = self._sift.detectAndCompute(template_image, None)
        self._image = template_image

        return template_image.shape[:2]

    def _shutdown_hook(self):
        """
        Executed upon a shutdown.
        """        
        rospy.loginfo("Shutting down robot brain!")
        
    def update(self, msg):
        try:
            cv_image: np.ndarray = self._cv_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

        except Exception as e:
            rospy.logerr(e)

        try: 
            # Convert the image to grayscale, then use SIFT to grab its key points.
            grayframe = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            kp_grayframe, desc_grayframe = self._sift.detectAndCompute(grayframe, None)

            # Match the key points to the template image.
            matches = self._flann.knnMatch(self._image_descriptors, desc_grayframe, k=2)
            
            # Put the good matches into a list of good points
            good_points = [m for m, n in matches if m.distance < 0.6 * n.distance]
            self._good_points_publisber.publish(Int16(len(good_points)))
            
            # Draw lines between the matched good points on the template image and camera image.
            frame = cv2.drawMatches(self._image, self._image_keypoints, grayframe, kp_grayframe, good_points, grayframe)

            # If there's enough good points, draw the homography as lines on the image.
            if len(good_points) > 30:
                query_pts = np.float32([self._image_keypoints[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
                train_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
                matrix, _ = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
                            
                h, w = self._image.shape
                pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, matrix)

                # We need to add the width of self._image to offset where the lines 
                # get drawn to compensate for self._image being drawn to the left!
                dst[:, :, 0] += w

                frame = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3)

                try:
                    self._extracted_image = self._update_extracted_image(cv_image, matrix)

                except Exception as e:
                    rospy.logerr(e)

            self._vision = frame
            
        except Exception as e:
            rospy.logerr(e)
    
    def _update_extracted_image(self, cv_image, matrix):
        h, w = 834, 1390
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)

        # Transform these points using the homography matrix to find the coordinates in the input image
        transformed_corners = cv2.perspectiveTransform(corners, matrix)

        # Calculate the bounding box of the transformed area
        [x_min, y_min] = np.int32(transformed_corners.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(transformed_corners.max(axis=0).ravel() + 0.5)

        # Define the width and height of the output image based on the transformed region
        width = x_max - x_min
        height = y_max - y_min

        # Define the destination points for a flat, rectangular output
        destination_corners = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])

        # Compute a new homography matrix for warping to the flattened view
        H_flattened, _ = cv2.findHomography(transformed_corners, destination_corners)

        # Warp the perspective to get the flattened image
        flattened_flag = cv2.warpPerspective(cv_image, H_flattened, (width, height))


        return flattened_flag


if __name__ == "__main__":
    rospy.init_node('robot_brain')

    time.sleep(5)   # Wait a few seconds for the ROS master node to register this node before we start doing anything

    robot_brain_node = RobotBrainNode()

    # This blocks until the node gets a shutdown signal, so that it continues to run 
    # endlessly. Otherwise, it would just exist here and the drone would go brain-dead.
    rospy.spin()