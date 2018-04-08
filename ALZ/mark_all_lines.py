import cv2
import numpy as np
# import matplotlib.pyplot as plt
import sys #for reading command line arguments etc.
import glob
# import matplotlib.image as mpimg
import math

def canny():
    import cv2
    import numpy as np
    from matplotlib import pyplot as plt

    # img = cv2.imread('road1.jpg')

    # Canny(image, min value, max value, ... aperature size, ...)
    edges = cv2.Canny(img, 100, 200)
    print "edges: ", edges
    plt.subplot(50), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(edges, cmap='gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

    plt.show()

def camera_get(img):
    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame
        cv2.imshow('frame', img)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

def hough_linesP(img):
    # hough lines ** probabilistic **:
    # more efficient, gives you the begining and ending of the lines:
    # give me an image object, I will find and draw linesing on it 

    import cv2
    import numpy as np


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200, apertureSize=3)


    #min line length, max line gap?
    lines = cv2.HoughLinesP(edges,rho = 1,theta = 1*np.pi/180,threshold = 100,minLineLength = 100,maxLineGap = 50)

    if lines is None:
        print " . . . . . no lines were found."
        return img 
    
    print "---------------------"
    #draw all the lines Hough found:
    # print "len(lines): ", len(lines), "lines: "
    # print lines
    # return img

    for line in lines:
        print ". . . . . . . . . . . . . . ."
        # line is 'normally' a list of one: [[x1, y1, x2, y2]] line[0] has the two points of the line

        x1 = line[0][0]
        y1 = line[0][1]
        x2 = line[0][2]
        y2 = line[0][3]
        pass
        # print "(", x1, ", ", y1, "), (", x2, ", ", y2, ")"
        line_length_squared = math.sqrt((x2 - x1)*(x2 - x1) + (y2 - y1)*(y2 - y1))
        line_slope = (y2 - y1)/float(x2 - x1)
        print "--------line: ", line, "d: : ", line_length_squared, "slope: ", line_slope
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return img

def hough_line(img):
    # give me an image object, I will find and draw linesing on it 

    import cv2
    import numpy as np


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 200, 350, apertureSize=3)


    #min line length, max line gap?
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    if lines is None:
        print " . . . . . no lines were found."
        return img 
    
    print "---------------------"
    #draw all the lines Hough found:
    # print "len(lines): ", len(lines), "lines: "
    # print lines
    # return img

    for line in lines:
        print ". . . . . . . . . . . . . . ."
        print "line: ", line

        for rho, theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            print "(", x1, ", ", y1, "), (", x2, ", ", y2, ")"
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return img

def hough():
    import cv2
    import numpy as np

    img = cv2.imread('road1.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200, apertureSize=3)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    for rho, theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        print "(", x1, ", ", y1, "), (", x2, ", ", y2, ")"
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imwrite('houghlines3.jpg', img)


    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def test_line_one_video():
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

        gray = hough_line(frame)

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

def test_on_images():
    #give this function the source, the destination 
    #   and the function to run on the images
    source_path = 'test_images/source_images/'
    paths = glob.glob(source_path + "*.jpg")
    print "Paths: ", paths
    for i, image_path in enumerate(paths):
        print " --- ", image_path, "----------------------"
        image = cv2.imread(image_path)
        result = hough_linesP(image)

        # plt.subplot(2, 3, i + 1)
        # plt.imshow(result)
        # mpimg.imsave('test_images/marked/' + image_path[12:-4] + '_detected.jpg', result)
        print 'test_images/marked/' + image_path[len(source_path):-4]
        cv2.imwrite('test_images/marked/' + image_path[len(source_path):-4] + '_detected.jpg', result)
    print ". . . . . . . . . . . . . . . . . . . . ."
    print "Batch processed. "

# canny()
# hough()

def main():
    print "------------------------------------------------"
    print "number of command line args: ", len(sys.argv)
    print "------------------------------------------------"
    print ""
    if len(sys.argv) >1:
        print "here is the commandline arguments: "
        # argv[0] is the name of this file
        # the rest are the command line arguments:
        for i in range(len(sys.argv)):
            print  sys.argv[i]
        print "------------------------------------------------" 
    print ""
    print ""
    test_line_one_video()
    # test_on_images()

if __name__ == '__main__':
    main()
