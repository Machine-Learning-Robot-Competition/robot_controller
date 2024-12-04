import cv2
import numpy as np
from typing import List, Tuple


ColorImage = np.ndarray
FlatImage = np.ndarray

LOWER_BLUE = lambda lower_saturation: np.array([110, lower_saturation, 50])
UPPER_BLUE = np.array([130, 255, 255])


def extract_blue(image: ColorImage, lower_saturation: int = 100) -> FlatImage:
    """
    Extract only the blue parts of an image (value and saturation compensated)
    
    Returns an matrix-like with the same (x, y) but single channel 
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Threshold the HSV image to get only blue colors
    return cv2.inRange(hsv_image, LOWER_BLUE(lower_saturation), UPPER_BLUE)


def order_contours_natural_reading(contours):
    """
    Order contours in natural reading order (top-down, then left-right).
    
    :param contours: List of contours from cv2.findContours()
    :return: Ordered list of contours
    """
    # Calculate the bounding box for each contour
    def get_contour_bounds(contour):
        x, y, w, h = cv2.boundingRect(contour)
        return (y, x)  # Prioritize y (vertical position) first
    
    # Sort contours first by y-coordinate (top), then by x-coordinate (left)
    return sorted(contours, key=get_contour_bounds)


def order_contours_natural_reading_with_details(contours):
    """
    Order contours in natural reading order with additional information.
    
    :param contours: List of contours from cv2.findContours()
    :return: Tuple of (ordered_contours, bounding_boxes)
    """
    # Calculate bounding boxes with more details
    def get_contour_details(contour):
        x, y, w, h = cv2.boundingRect(contour)
        return {
            'contour': contour,
            'bounds': (y, x),
            'center': (x + w // 2, y + h // 2),
            'area': cv2.contourArea(contour)
        }
    
    # Process contours with additional metadata
    contour_details = [get_contour_details(cnt) for cnt in contours]
    
    # Sort by vertical position (y), then horizontal position (x)
    sorted_details = sorted(contour_details, key=lambda x: x['bounds'])
    
    # Return ordered contours and their details
    return (
        [detail['contour'] for detail in sorted_details],
        sorted_details
    )


def extract_contours(image: FlatImage) -> List[np.ndarray]:
    """
    Extract the valid contours from an image, ordered in typical reading order (ie. left-right, then top-down)

    A valid contour is a non-internal contour (like the contour around the internal space of the letter 'A') and one that
    does not like on the boundary of the image.
    """
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # I don't like this either
    try:
        height, width, _ = image.shape
    except Exception:
        height, width = image.shape

    filtered_contours = []

    for i, contour in enumerate(contours):
        # If we are looking at an inner contour (like the bound ofinterior enclosed empty space of the letter A)
        # we want to skip it. hierarchy[0][i][3] == -1 indicates that the contour i is not a child contour.
        if hierarchy[0][i][3] == -1:
            x, y, w, h = cv2.boundingRect(contour)

            # Only append contours that don't borders the edge (which can erroneously arise due tonoise due to the quadrilateral detection)
            if x > 0 and y > 0 and (x + w) < width and (y + h) < height:
                filtered_contours.append(contour)

    ordered_contours = order_contours_natural_reading(filtered_contours)

    return ordered_contours


def extract_letters(image: FlatImage, contours: List[np.ndarray]) -> List[FlatImage]:
    """
    Extract the letters from an image by cropping out the contours. 
    """
    processed_letters = []

    for i, contour in enumerate(contours):
        # Get bounding box and crop the contour
        x, y, w, h = cv2.boundingRect(contour)
        cropped = image[y:y+h, x:x+w]  # Crop the bounding box region

        # Store the processed letter
        processed_letters.append(cropped)

    return processed_letters


def resize_if_bigger(image, target_width, target_height):
    # Get the original dimensions of the image
    original_height, original_width = image.shape[:2]
    
    # Determine the new dimensions
    new_width = min(original_width, target_width)
    new_height = min(original_height, target_height)
    
    # Check which dimension(s) need to be resized
    if original_width > target_width and original_height > target_height:
        # Resize both dimensions, preserving aspect ratio
        aspect_ratio = original_width / original_height
        if new_width / new_height > aspect_ratio:
            new_width = int(new_height * aspect_ratio)
        else:
            new_height = int(new_width / aspect_ratio)
    elif original_width > target_width:
        # Resize width only
        new_height = original_height
    elif original_height > target_height:
        # Resize height only
        new_width = original_width
    
    # Resize the image using cv2.resize
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    return resized_image


def pad_to_size(array, target_shape=(40, 40), fill_value=0) -> FlatImage:
    """
    Pad a 2D NumPy array to a target shape, centering the original array.
    
    Parameters:
    -----------
    array : numpy.ndarray
        Input 2D array to be padded
    target_shape : tuple, optional
        Desired output shape (default is (40, 40))
    fill_value : numeric, optional
        Value to use for padding (default is 0)
    
    Returns:
    --------
    numpy.ndarray
        Padded array with shape equal to target_shape
    
    Raises:
    -------
    ValueError
        If input array is larger than target shape in either dimension
    """
    # Input validation
    if array.ndim != 2:
        raise ValueError("Input must be a 2D array")
    
    if array.shape[0] > target_shape[0] or array.shape[1] > target_shape[1]:
        raise ValueError(f"Input array {array.shape} is larger than target shape {target_shape}")
    
    # Calculate padding
    pad_height = target_shape[0] - array.shape[0]
    pad_width = target_shape[1] - array.shape[1]
    
    # Calculate padding for top/bottom and left/right
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    
    # Create padded array
    padded_array = np.full(target_shape, fill_value, dtype=array.dtype)
    
    # Insert original array into center of padded array
    padded_array[pad_top:pad_top+array.shape[0], 
                 pad_left:pad_left+array.shape[1]] = array
    
    return padded_array


def pad_image_collection(images, target_shape=(40, 40), fill_value=0) -> List[FlatImage]:
    """
    Apply padding to a collection of images.
    
    Parameters:
    -----------
    images : list or numpy.ndarray
        Collection of 2D arrays to be padded
    target_shape : tuple, optional
        Desired output shape for each image
    fill_value : numeric, optional
        Value to use for padding
    
    Returns:
    --------
    numpy.ndarray
        Array of padded images
    """
    return np.array([pad_to_size(resize_if_bigger(img, target_shape[1], target_shape[0]), target_shape, fill_value) for img in images])


def bbox_min_distance(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """
    Calculate the minimum distance between two bounding boxes.

    Calculates distance based on the closest points between the boxes.
    If boxes overlap, returns 0.
    
    :param np.ndarray bbox1: First bounding box in format [x1, y1, w1, h1]
    :param np.ndarray bbox2: Second bounding box in format [x2 y2, w2, h2]
    :returns: Minimum distance between the bounding boxes
    """
    # Unpack bounding boxes
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    # Calculate box corners
    box1_left, box1_right = x1, x1 + w1
    box1_top, box1_bottom = y1, y1 + h1
    
    box2_left, box2_right = x2, x2 + w2
    box2_top, box2_bottom = y2, y2 + h2
    
    # Check for overlap
    if (box1_left <= box2_right and box1_right >= box2_left and
        box1_top <= box2_bottom and box1_bottom >= box2_top):
        return 0.0
    
    # Calculate horizontal distance
    if box1_right < box2_left:
        h_dist = box2_left - box1_right
    elif box2_right < box1_left:
        h_dist = box1_left - box2_right
    else:
        h_dist = 0
    
    # Calculate vertical distance
    if box1_bottom < box2_top:
        v_dist = box2_top - box1_bottom
    elif box2_bottom < box1_top:
        v_dist = box1_top - box2_bottom
    else:
        v_dist = 0
    
    # Return Euclidean distance
    return np.sqrt(h_dist**2 + v_dist**2)


def identify_spaces(contours: List[np.ndarray], minimum_distance: float = 20.0) -> List[Tuple[int, int]]:
    """
    Determine whether it appears that a set of contours are seperated into words. 
    Returns the indices in the sequence of contours that seperation between adjacent contours occur. 
    
    For example, if contours representing the string "CRIMEFAILEDPRANK" and it is found that
    there is a seperation greater than ``minimum_distance`` between CRIME, FAILED, and PRANK, the
    returned value would be [(4, 5), (10, 11)].

    :param List[np.ndarray] contours: sequence of contours that will be examined 
    :param float minimum_distance: minimum distance in pixels that adjacent contours must be seperated by to identify a space
    """
    # Loop through the contours to find bounding boxes and distances between them
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]

    # Detect spaces (both horizontal and vertical)
    spaces = []

    for i in range(len(bounding_boxes) - 1):
        bbox1 = bounding_boxes[i]
        bbox2 = bounding_boxes[i + 1]
            
        # Check vertical distance (check if the y-coordinates of the bounding boxes are far apart)
        vertical_distance = bbox_min_distance(bbox1, bbox2)
        if vertical_distance > minimum_distance:
            spaces.append((i, i + 1))

    return spaces


def collect_words(letters: List[FlatImage], spaces: List[Tuple[int, int]]) -> List[List[FlatImage]]:
    """
    Collect words from a list of letters and the spaces between them.
    """
    words = []

    prev_index = 0
    for space in spaces:
        words.append(letters[prev_index:space[1]])
        prev_index = space[1]

    words.append(letters[prev_index:])

    return words
