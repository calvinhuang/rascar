import cv2
import numpy as np


def adaptive_threshold(img):
    """
    Takes in an image and return the lower and higher thresholds of the image
    :param img: Image to find threshold of
    :return: Lower threshold, higher threshold
    """

    sigma = 0.33
    v = np.median(img)

    lower = int(max(0, (1.0 - sigma) * v))
    higher = int(min(255, (1.0 + sigma) * v))

    return lower, higher


def canny_otsu(image):
    """
    Takes in an image, applies a Gaussian blur, threshold, Canny
    :param image: Image
    :return:
    """
    gauss = cv2.GaussianBlur(image, (5, 5), 0)
    _, thresh = cv2.threshold(gauss, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    canny_thresh = cv2.Canny(thresh, 100, 200)

    return canny_thresh
