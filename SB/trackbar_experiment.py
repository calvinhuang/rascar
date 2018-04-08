import cv2
import numpy as np


class trackbars:
    def __init__(self, name_list, max_list, call_back_list):
        self._img = np.zeros((1,512,3), np.uint8)
        cv2.namedWindow('image')
        cv2.imshow('image',self._img)

        for i in range(0,len(name_list)):
            cv2.createTrackbar(name_list[i],'image', max_list[i]/2, max_list[i], call_back_list[i])
    def _pos(self, name):
        return cv2.getTrackbarPos(name, 'image')


def nothing(x):
    # print x   # x is the current value of the trackbar triggering this function
    pass

# # Create a black image, a window
# img = np.zeros((300,512,3), np.uint8)
# cv2.namedWindow('image')
#
# # create trackbars for color change
# # trackbar name, window name, value, max, call back function | min always 0
# cv2.createTrackbar('R','image',120,255,nothing)
# cv2.createTrackbar('G','image',150,255,nothing)
# cv2.createTrackbar('B','image',70,255,nothing)
#
# # create switch for ON/OFF functionality
# switch = '0 : OFF \n1 : ON'
# cv2.createTrackbar(switch, 'image',1,1,nothing)

t = trackbars(['R','G', 'B'],[255,255,255],[nothing, nothing, nothing])

img2 = cv2.imread("test_images/solidWhiteCurve.jpg")
while(1):
    # cv2.imshow('image',img)
    cv2.imshow('image2', img2)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    # get current positions of four trackbars
    # trackbar name, window name
    r = t._pos('R') #cv2.getTrackbarPos('R','image')
    g = t._pos('G') #cv2.getTrackbarPos('G','image')
    b = t._pos('B') #cv2.getTrackbarPos('B','image')
    # s = cv2.getTrackbarPos(switch,'image')
    #
    # if s == 0:
    #     img[:] = 0
    # else:
    #     img[:] = [b,g,r]
    # print r, g, b
    # t._img[:] = [b,g,r]
cv2.destroyAllWindows()