import numpy as np
import cv2
#http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html#display-video

#cap is a VideoCapture object
#zero is a video file name or device index

def video():
    cap = cv2.VideoCapture(0)

    #sometimes cap is not opened (?) use cap.open() to open it
    if not cap.isOpened():
        cap.open()
    while(True):
        # Capture frame-by-frame
        #if read returns false, that's the end of the video.
        ret, frame = cap.read()

        # Our operations on the frame come here:
        # cvtColor will return an image that gets stroed in gray

        #------------------------- Video Features ------------------
        #acess video features: cap.get(propId)
        #frame width and height: props 3 and 4:
        # print "frame size: ", cap.get(3), 'X', cap.get(4)
        #-----------------------------------------------------------
        # set features by cap.set(propId, value):

        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # # gray = frame
        # # see img_proc_canny_edge_detection.py:
        # edges = cv2.Canny(gray, 100, 200)
        # gray = cv2.add(gray, edges)

        gray = mark_lanes(frame)

        # Display the resulting frame
        cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        elif cv2.waitKey(1) & 0XFF == ord('1'):
            ret = cap.set(3, cap.get(3)/2)
            ret = cap.set(4, cap.get(4)/2)
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
