import cv2


def fast(img):
    """
    Uses the FAST algorithm to track objects
    :param img: Image to track objects
    :return: image with points drawn, Key points
    """
    fast_t = cv2.FastFeatureDetector_create()

    points = fast_t.detect(img, None)
    kimg = cv2.drawKeypoints(img, points, None, color=(0, 255, 0))
    
    return kimg, points


def orb(img):
    """
    Uses the ORB alogrithm to track objects,
    :param img: Image to track objects
    :return: Image with points drawn, Key points
    """
    orb_t = cv2.ORB_create()

    points = orb_t.detect(img, None)
    points, des = orb_t.compute(img, points)

    kimg = cv2.drawKeypoints(img, points, None, (0, 255, 0), 0)

    return kimg, points
