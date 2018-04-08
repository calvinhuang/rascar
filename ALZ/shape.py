import numpy as np
import cv2

#black image
img = np.zeros((512,512,3))

#draw line
cv2.line(img, (0,0), (511,511), (255,0,0), 5)

#rectangle
cv2.rectangle(img,(384,0),(510,128),(0,255,0),3)

#circle
cv2.circle(img,(447,63), 63, (0,0,255), -1)

#Ellipse
cv2.ellipse(img,(256,256), (50,50), 0, -30, 240, 255, -1)
cv2.circle(img,(256,256), 20, (0,0,0), -1)

#polygon
pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
pts = pts.reshape((-1,1,2))
cv2.polylines(img,[pts],True,(0,255,255))

#text
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,'OpenCV',(10,500), font, 4,(255,255,255),2,cv2.LINE_AA)

#showing function
cv2.imshow('image', img)
cv2.waitKey(0) & 0xFF
if k == 27:     #27 == ESC key
    cv2.destroyAllWindow()
elif k == ord('s'): #save
    cv2.imwrite('new_image.jpg', img)
    cv2.destroyAllWindow()