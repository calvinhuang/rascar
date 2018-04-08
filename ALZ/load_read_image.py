import numpy as np
import cv2

#load a color image
img = cv2.imread('sample1.jpg', 1)

#cv2.nameWindow('img', cv2.WINDOWNORMAL)
cv2.imshow('image', img)
cv2.waitKey(0) & 0xFF
if k == 27:     #27 == ESC key
    cv2.destroyAllWindow()
elif k == ord('s'): #save
    cv2.imwrite('new_image.jpg', img)
    cv2.destroyAllWindow()
