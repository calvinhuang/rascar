import cv2
import numpy as np
import hough

def contours(img, dest=None):
    if dest is None:
        dest = img

    _, thresh = cv2.threshold(img, 127, 255, 0)

    img2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(dest, contours, -1, (0, 255, 0), 3)

    return dest


if __name__ == '__main__':
    vid = cv2.VideoCapture(0)

    while vid.isOpened():
        ret, frame = vid.read()

        e = hough.canny(frame)

        r = hough.roi(e)

        c = contours(r, frame)

        cv2.imshow('v', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

vid.release()
cv2.destroyAllWindows()
