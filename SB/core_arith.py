# http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_core/py_image_arithmetics/py_image_arithmetics.html#image-arithmetics

# Note There is a difference between OpenCV addition and Numpy addition.
# OpenCV addition is a saturated operation while Numpy addition is a modulo operation.


import cv2
import numpy as np
def add():
    x = np.uint8([250])
    y = np.uint8([10])
    print 'X: ', x, ', Y: '
    print 'add(x, y): ', cv2.add(x, y) # 250+10 = 260 => 255
    print 'x + y: ', x + y   # 250+10 = 260 % 256 = 4

    img1 = cv2.imread("1.jpg")


    img2 = cv2.imread("2.jpg")

    print "========================================================================"
    print "shape: ", img1.shape, ", size: ", img1.size, ", type: ", img1.dtype
    print "shape: ", img2.shape, ", size: ", img2.size, ", type: ", img2.dtype
    print "========================================================================"

    # dst = cv2.add(img1[0:0,300:300], img2[0:0,300:300])
    # dst = cv2.addWeighted(img1[0:0,300:300], 0.7, img2[0:0,300:300], 0.3, 0)
    dst = cv2.addWeighted(img1[0:1000,000:1000], 0.9, img2[0:1000,0:1000], 0.1, 0)
    # dst = cv2.addWeighted(img1, 0.7, img2, 0.3, 0)
    img1[0:1000, 0:1000] = dst
    cv2.imshow("xxx", img1)
    # cv2.imshow("xxx", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def bitwise():
    # Load two images
    img1 = cv2.imread('road1.jpg')
    img2 = cv2.imread('road2.jpg')

    # I want to put logo on top-left corner, So I create a ROI
    rows, cols, channels = img2.shape
    roi = img1[0:rows, 0:cols]

    # Now create a mask of logo and create its inverse mask also
    img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # Now black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(img2, img2, mask=mask)

    # Put logo in ROI and modify the main image
    dst = cv2.add(img1_bg, img2_fg)
    img1[0:rows, 0:cols] = dst

    cv2.imshow('res', img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


bitwise()
