import numpy as np
import cv2


# http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_drawing_functions/py_drawing_functions.html#drawing-functions


# img       : The image where you want to draw the shapes
# color     : Color of the shape. for BGR, pass it as a tuple, eg: (255,0,0) for blue. For grayscale, just pass the scalar value.
# thickness : Thickness of the line or circle etc. If -1 is passed for closed figures like circles, it will fill the shape. default thickness = 1
# lineType  : Type of line, whether 8-connected, anti-aliased line etc. By default, it is 8-connected. cv2.LINE_AA gives anti-aliased line which looks great for curves.





# Create a black image
img = np.zeros((512,1000,3), np.uint8)

# Draw a diagonal blue line with thickness of 5 px
#   image object, starting and ending coords, color, thickness:
cv2.line(img,(0,0),(511,511),(255,0,0),5)


cv2.rectangle(img,(384,0),(510,128),(0,255,0),3)


cv2.circle(img,(447,63), 63, (0,0,255), -1)

# ellipese: second and third args: major and minor axis:
#   To draw the ellipse, we need to pass several arguments.
#       One argument is the center location (x,y).
#   Next argument is axes lengths (major axis length, minor axis length).
#   angle is the angle of rotation of ellipse in anti-clockwise direction.
#   startAngle and endAngle denotes the starting and ending
#           of ellipse arc measured in clockwise direction from major axis.
#           i.e. giving values 0 and 360 gives the full ellipse.
#           For more details, check the documentation of cv2.ellipse().


cv2.ellipse(img,(256,256),(100,50),0,0,360,255,-1)

#polygons:
pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
pts = pts.reshape((-1,1,2))
#if the third argument is false, lines will connect the points
cv2.polylines(img,[pts],True,(0,255,255))


#text:
# Text data that you want to write
# Position coordinates of where you want put it (i.e. bottom-left corner where data starts).
# Font type (Check cv2.putText() docs for supported fonts)
# Font Scale (specifies the size of font)
# regular things like color, thickness, lineType etc. For better look, lineType = cv2.LINE_AA is recommended.

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,'Y!',(10,500), font, 0.5,(255,255,255),2,cv2.LINE_AA)
cv2.putText(img,'X!',(100,500), font, 0.2,(255,255,255),2,cv2.LINE_AA)
cv2.putText(img,'Z!',(200,500), font, .7,(255,255,255),2,cv2.LINE_AA)

cv2.imshow("drawings", img)

while (True):
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
