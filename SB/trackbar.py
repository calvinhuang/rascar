import cv2
import numpy as np

class trackbars:
    def __init__(self, name_list, max_list, call_back_list=None):
        self._img = np.zeros((1,512,3), np.uint8)
        cv2.namedWindow('image')
        cv2.imshow('image',self._img)

        for i in range(0,len(name_list)):
            if call_back_list is None:
                cv2.createTrackbar(name_list[i], 'image', max_list[i]/2, max_list[i], self._nothing)
            else:
                cv2.createTrackbar(name_list[i], 'image', max_list[i]/2, max_list[i], call_back_list[i])

    def _pos(self, name):
        return cv2.getTrackbarPos(name, 'image')

    def _nothing(self, x):
        pass

    def _run(self):
        while (1):

            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break

        cv2.destroyAllWindows()