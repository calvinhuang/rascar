import numpy as np
import cv2


import numpy as np
import cv2

img = cv2.imread('fox.jpg',0)
cv2.imshow('image',img)
k = cv2.waitKey(0)  #areg 0: stay indefinitely.
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    cv2.imwrite('fox.png',img)
    cv2.destroyAllWindows()
