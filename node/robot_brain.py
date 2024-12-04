#! /usr/bin/env python3

import tensorflow as tf
import cv2
import threading
import rospy
import time
import pathlib
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Int16, Empty, Bool, String
from cv_bridge import CvBridge
from cv_utils import time_it, extract_blue, extract_contours, extract_letters, pad_to_size, pad_image_collection, identify_spaces, collect_words, ColorImage, FlatImage
import traceback
from typing import List, Tuple
import actionlib
from robot_controller.msg import ReadClueboardAction, ReadClueboardFeedback, ReadClueboardResult, ReadClueboardGoal
import Levenshtein
import pathlib


REFERENCE_IMAGE_PATH: str = str(pathlib.Path(__file__).absolute().parent.parent / "media" / "reference_image_cropped.png")  # Update REFERENCE_IMAGE_DIMENSIONS!
CAMERA_FEED_TOPIC: str = "/front_cam/camera/image"
RESULT_PUBLISH_TOPIC: str = "/robot_brain/vision"
GOOD_POINTS_TOPIC: str = "/robot_brain/good_points"
EXTRACTED_IMAGE_TOPIC: str = "/robot_brain/extracted_image"
LETTERS_PUBLISH_TOPIC: str = "/robot_brain/letters_image"
SCORE_TRACKER_TOPIC = "/score_tracker"
REFERENCE_IMAGE_DIMENSIONS = (804, 1178)
TARGET_SHAPE = (48, 42)
OUTPUT_DIMS = (int(360*1.5), int(210*1.5))  # Height, Width, 315, 540
MAX_ATTEMPTS = 30
DEBUG = True

encoding = {
    "A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7, "I": 8, "J": 9,
    "K": 10, "L": 11, "M": 12, "N": 13, "O": 14, "P": 15, "Q": 16, "R": 17, "S": 18,
    "T": 19, "U": 20, "V": 21, "W": 22, "X": 23, "Y": 24, "Z": 25, "0": 26, "1": 27,
    "2": 28, "3": 29, "4": 30, "5": 31, "6": 32, "7": 33, "8": 34, "9": 35
}

clue_types = {
    "SIZE": "1",
    "VICTIM": "2",
    "CRIME": "3",
    "TIME": "4",
    "PLACE": "5",
    "MOTIVE": "6",
    "WEAPON": "7",
    "BANDIT": "8"
}

reversed_encoding = {value: key for key, value in encoding.items()}

MODEL_PATH = str(pathlib.Path(__file__).absolute().parent.parent / "models" / "27ijcbd.keras")
DO_CLASSIFIER_PATH = str(pathlib.Path(__file__).absolute().parent.parent / "models" / "DO_classifier.keras")
B9_CLASSIFIER_PATH = str(pathlib.Path(__file__).absolute().parent.parent / "models" / "B9_classifier.keras")


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

        self._server = actionlib.SimpleActionServer(
            'read_clueboard', 
            ReadClueboardAction, 
            self._read_clueboard, 
            False
        )
        self._server.start()

        self._vision_publisber = rospy.Publisher(RESULT_PUBLISH_TOPIC, Image, queue_size=1)
        self._letters_publisher = rospy.Publisher(LETTERS_PUBLISH_TOPIC, Image, queue_size=1)
        self._extracted_image_publisher = rospy.Publisher(EXTRACTED_IMAGE_TOPIC, Image, queue_size=1)
        self._good_points_publisber = rospy.Publisher(GOOD_POINTS_TOPIC, Int16, queue_size=1)
        self._score_tracker_publisher = rospy.Publisher(SCORE_TRACKER_TOPIC, String, queue_size=10)

        self._model = tf.keras.models.load_model(MODEL_PATH)
        self._do_model = tf.keras.models.load_model(DO_CLASSIFIER_PATH)
        self._b9_model = tf.keras.models.load_model(B9_CLASSIFIER_PATH)

        rospy.loginfo("Brain Initialized!")

    def _get_closest_word(self, cnn_word):
        closest_word = None
        smallest_distance = float('inf')

        for word in clue_types.keys():
            distance = Levenshtein.distance(cnn_word, word)
            if distance < smallest_distance:
                smallest_distance = distance
                closest_word = word
        
        return closest_word

    def _read_clueboard(self, goal: ReadClueboardGoal):
        feedback = ReadClueboardFeedback()
        result = ReadClueboardResult()

        success, padded_letters, split_contours = self.see()
        rospy.loginfo(f"Contours: {split_contours}")
        feedback.clueboard_lock_success = success
        self._server.publish_feedback(feedback)

        if success:
            clue_type, clue_value = self.read(padded_letters, split_contours)
            rospy.loginfo(f"Setting result to: {clue_type}: {clue_value}")

            clue_type_verified = self._get_closest_word(clue_type)
            clue_id = clue_types[clue_type_verified]

            result.clueboard_text = [clue_type_verified, clue_value]

            self._server.set_succeeded(result)

            self._score_tracker_publisher.publish(String(f"Team4,password,{clue_id},{clue_value}"))

        else:
            rospy.loginfo("Aborting clueboard read!")
            self._server.set_aborted(result)
    
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

    def process_image(self, cv_image: np.ndarray, image: np.ndarray, image_keypoints, image_descriptors):
        # Convert the image to grayscale, then use SIFT to grab its key points.
        grayframe = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        kp_grayframe, desc_grayframe = self._sift.detectAndCompute(grayframe, None)

        # Match the key points to the template image.
        matches = self._flann.knnMatch(image_descriptors, desc_grayframe, k=2)
        
        # Put the good matches into a list of good points
        good_points = [m for m, n in matches if m.distance < 0.65 * n.distance]
        
        # Draw lines between the matched good points on the template image and camera image.
        frame = cv2.drawMatches(image, image_keypoints, grayframe, kp_grayframe, good_points, grayframe)

        return good_points, frame, kp_grayframe
    
    @staticmethod
    def get_homography(good_points, kp_grayframe, image: np.ndarray, image_keypoints, frame):
        query_pts = np.float32([image_keypoints[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
        train_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
        matrix, _ = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
        
        try:
            h, w = image.shape
        except:
            h, w, _ = image.shape

        pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, matrix)

        # We need to add the width of self._image to offset where the lines 
        # get drawn to compensate for self._image being drawn to the left!
        dst[:, :, 0] += w

        frame = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3)

        return frame, matrix        

    def see(self):
        attempts = 0
        while attempts < MAX_ATTEMPTS:
            rospy.loginfo(f"Beginning attempt at getting homography {attempts}...")

            try:
                msg = rospy.wait_for_message(CAMERA_FEED_TOPIC, Image, timeout=5)
                cv_image: np.ndarray = self._cv_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

                good_points, frame, kp_grayframe = self.process_image(cv_image, self._image, self._image_keypoints, self._image_descriptors)
                self._good_points_publisber.publish(Int16(len(good_points)))

                if len(good_points) > 15:
                    frame, matrix = self.get_homography(good_points, kp_grayframe, self._image, self._image_keypoints, frame)

                    extracted_image, success = self._update_extracted_image(cv_image, matrix)

                    if success:
                        extracted_image_msg = self._cv_bridge.cv2_to_imgmsg(extracted_image)
                        self._extracted_image_publisher.publish(extracted_image_msg)

                        rospy.loginfo("Successfully acquired homography!")
                        break

                vision_msg = self._cv_bridge.cv2_to_imgmsg(frame)
                self._vision_publisber.publish(vision_msg)

            except Exception as e:
                rospy.logerr(f"{e} {traceback.format_exc()}")
                return False, None, None
            
        else:
            rospy.logerr(f"Failed to find homography after {MAX_ATTEMPTS} attempts!")
            return False, None, None

        try:
            blue_image: FlatImage = extract_blue(extracted_image)
            if DEBUG: 
                cv2.imwrite("./blue_image.png", blue_image)

            contours, split_contours = extract_contours(blue_image)
            rospy.loginfo(f"Found contours: {len(contours)}")
            if DEBUG:
                output_image = extracted_image.copy()
                cv2.drawContours(output_image, contours, -1, (0, 255, 0), 2)  
                cv2.imwrite("./output_image.png", output_image)

            letters: List[FlatImage] = extract_letters(blue_image, contours)
                
            if DEBUG:
                for i, letter in enumerate(letters):
                    rospy.loginfo(f"Size {i}: {letter.shape}")
                    cv2.imwrite(f"./letters/{i}.png", letter)

            try:
                padded_letters: List[FlatImage] = pad_image_collection(letters, target_shape=TARGET_SHAPE)

                if DEBUG:
                    for i, padded_letter in enumerate(padded_letters):
                        cv2.imwrite(f"./padded_letters/{i}.png", padded_letter)

                # spaces: List[Tuple[int, int]] = identify_spaces(contours, minimum_distance=30.0)

                if DEBUG:
                    banner_image = np.hstack(padded_letters)
                    cv2.imwrite("./banner_image.png", banner_image)

                return True, padded_letters, split_contours

            except Exception as e:
                rospy.logerr(f"{e} {traceback.format_exc()}")

            letters_msg = self._cv_bridge.cv2_to_imgmsg(output_image)
            self._letters_publisher.publish(letters_msg)

        except Exception as e:
            rospy.logerr(f"{e} {traceback.format_exc()}")
            return False, None, None

    def read(self, letters, split_contours):
        word_images = [[], []]
        for letter_index in split_contours[0]:
            word_images[0].append(letters[letter_index])

        for letter_index in split_contours[1]:
            word_images[1].append(letters[letter_index])

        # word_images: List[List[FlatImage]] = collect_words(letters, spaces)

        try:
            words: List[str] = [''.join([self.predict(letter) for letter in word]) for word in word_images]
            clue_type = words[0]
            clue_value = ' '.join(words[1:])
        except Exception as e:
            rospy.logerr(f"{e} {traceback.format_exc()}")
        
        return clue_type, clue_value
        
    
    def predict(self, X_data):
        reshaped = X_data.reshape(*TARGET_SHAPE)
        input_data = np.expand_dims(reshaped, axis=0)
        result = self._model.predict(input_data)
        prediction = reversed_encoding[np.argmax(result[0])]

        if prediction == "D" or prediction == "O":
            rospy.loginfo(f"Normal model says: {prediction}")
            rospy.loginfo("Using DO model...")

            result = np.sum(X_data[4, 6])
            rospy.loginfo(f"DO Model Result: {result}")
            if result > 128:
                prediction = "D"
                rospy.loginfo("DO model says D")
            else:
                prediction = "O"
                rospy.loginfo("DO model says O")

        if prediction == "O":
            result = np.sum(X_data[20, 20])
            rospy.loginfo(f"O-6 Model Result: {result}")
            if result > 128:
                prediction = "6"
                rospy.loginfo("O-6 model says 6")
            else:
                prediction = "O"
                rospy.loginfo("O-6 model says O")
        
        # [46, 34]
        if prediction == "O":
            result = np.sum(X_data[46, 34])
            rospy.loginfo(f"O-Q Model Result: {result}")
            if result > 128:
                prediction = "Q"
                rospy.loginfo("O-Q model says 6Q")
            else:
                prediction = "O"
                rospy.loginfo("O-Q model says O")

        if prediction == "B" or prediction == "9":
            rospy.loginfo(f"Normal model says: {prediction}")
            rospy.loginfo("Using B9 model...")
            result = self._b9_model.predict(input_data)
            rospy.loginfo(f"B9 Model Result: {result}")
            if result < 0.5:
                prediction = "B"
                rospy.loginfo("B9 model says B")
            else:
                prediction = "9"
                rospy.loginfo("B9 model says 9")
            
        return prediction

    def _align_flag(self, image):
        cv2.imwrite("./raw_image.png", image)
        # Detect edges in the color image using pixel brightness
        blue_image: FlatImage = extract_blue(image, lower_saturation=50)
        edges = cv2.Canny(blue_image, 50, 150)     
        cv2.imwrite("./edges.png", edges)
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        output_image = image.copy()
        cv2.drawContours(output_image, contours, -1, (0, 255, 0), 2)
        cv2.imwrite("./quadrilateral.png", output_image)

        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        if len(approx) == 4:  # If a quadrilateral is detected
            # Reorder the points for perspective transform
            approx = order_points(approx.reshape(-1, 2))
            
            # Define the destination points (aligned rectangle)
            width = int(max(
                np.linalg.norm(approx[0] - approx[1]),
                np.linalg.norm(approx[2] - approx[3])
            ))
            height = int(max(
                np.linalg.norm(approx[1] - approx[2]),
                np.linalg.norm(approx[3] - approx[0])
            ))
            
            dst_corners = np.float32([
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1]
            ])
            
            # Compute the homography to align the flag
            H_align = cv2.getPerspectiveTransform(approx, dst_corners)
            
            # Warp the color image to align the flag
            aligned_flag = cv2.warpPerspective(image, H_align, (width, height))

            aligned_flag_resized = cv2.resize(aligned_flag, OUTPUT_DIMS)

            return aligned_flag_resized, True

        # Return the original image if no quadrilateral is found
        return image, False
    
    @time_it
    def _update_extracted_image(self, cv_image, matrix):
        success = False
        h, w = REFERENCE_IMAGE_DIMENSIONS
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

        try:
            flattened_flag, align_success = self._align_flag(flattened_flag)
            success = align_success

        except Exception as e:
            rospy.logerr(f"Error Aligning Image: {e} \n {traceback.format_exc()}")

        return flattened_flag, success


def order_points(pts):
    # Sort by x-coordinates
    x_sorted = pts[np.argsort(pts[:, 0]), :]
    
    # Split into left-most and right-most points
    left = x_sorted[:2, :]
    right = x_sorted[2:, :]
    
    # Order left-most points by y-coordinates
    left = left[np.argsort(left[:, 1]), :]
    (tl, bl) = left  # Top-left, Bottom-left
    
    # Order right-most points by y-coordinates
    right = right[np.argsort(right[:, 1]), :]
    (tr, br) = right  # Top-right, Bottom-right
    
    return np.array([tl, tr, br, bl], dtype="float32")


if __name__ == "__main__":
    rospy.init_node('robot_brain')

    time.sleep(3)   # Wait a few seconds for the ROS master node to register this node before we start doing anything

    robot_brain_node = RobotBrainNode()

    # This blocks until the node gets a shutdown signal, so that it continues to run 
    # endlessly. Otherwise, it would just exist here and the drone would go brain-dead.
    rospy.spin()
