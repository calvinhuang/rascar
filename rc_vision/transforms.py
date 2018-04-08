import cv2
import numpy as np


def warp(img, width, height):
    p1 = np.float32([[0, int(height/3)],
                     [width, int(height/3)],
                     [0, height],
                     [width, height]])

    p2 = np.float32([[0, 0],
                     [int(width*.9), 0],
                     [int(width*.25), int((height/3)*2.5)],
                     [int(width-(width*.25))-100, int((height/3)*2.5)]])

    transform = cv2.getPerspectiveTransform(p1, p2)

    warped = cv2.warpPerspective(img, transform, (int(width*.9), int((height/2)*2.5)))

    return warped

