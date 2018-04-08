import cv2
import numpy as np

#http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_mouse_handling/py_mouse_handling.html#mouse-handling


# mouse callback function
def draw_circle(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONUP:
        print "(", x, y, ")"
        cv2.circle(img,(x,y),5,(255,0,0),-1)


# List of events:
events = [i for i in dir(cv2) if 'EVENT' in i]
print events

# Create a black image, a window and bind the function to window
img = np.zeros((512,512,3), np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)

while(1):
    cv2.imshow('image',img)
    if cv2.waitKey(20) & 0xFF == 27:
        break
cv2.destroyAllWindows()