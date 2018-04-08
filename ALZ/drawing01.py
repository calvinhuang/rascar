import numpy as np
import cv2

#create a black image
img = np.zeros((512, 512, 3), np.uint8)

#draw a diagonal blue line with thickness of 5px
cv2.line(img, (10, 10), (400, 400), (255, 0, 0), 5)

#Draw a rectangle
cv2.rectangle(img, (300, 0), (600, 300), (0, 255, 0), 3)

#Draw circle
cv2.circle(img, (0, 0), 70, (0, 0, 255), -1)

#showing function
cv2.imshow('image', img)
cv2.waitKey(0) & 0xFF
if k == 27:     #27 == ESC key
    cv2.destroyAllWindow()
elif k == ord('s'): #save
    cv2.imwrite('new_image.jpg', img)
    cv2.destroyAllWindow()

