import numpy as np
from PyQt5 import QtGui 


# Source: stackoverflow.com/questions/34232632/
def convert_cv_to_pixmap(cv_img: np.ndarray) -> QtGui.QPixmap:
    """
    Convert an OpenCV-compatible ndarray to a Qt-compatible Pixmap.

    @param np.ndarray cv_img: OpenCV-compatible ndarray that will be converted.
    @return the `cv_img` converted to a Qt-compatible PixmapQt-compatible Pixmap.
    """
    height, width, channel = cv_img.shape
    bytesPerLine = channel * width
    q_img = QtGui.QImage(cv_img.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
    return QtGui.QPixmap.fromImage(q_img)