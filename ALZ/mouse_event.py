import cv2
import numpy as np

events = [i for i in dir(cv2) if 'EVENT' in i]
print events

# mouse callback function
def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img, (x,y), 100, (255, 0, 0), -1)

#window with a black pic
img = np.zeros((512, 512, 3), np.unit8)
cv2.nameWindow('image')
cv2.setMouseCallback