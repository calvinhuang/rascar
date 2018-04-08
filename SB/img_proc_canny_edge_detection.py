import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys #for reading command line arguments etc.
import glob
import matplotlib.image as mpimg
import math

from operator import itemgetter

class Text():
    def __init__(self):
        self.row = 0
        self.data = []
    def add(self, s):
        self.data.append(s)
        self.row = self.row + 1
    def draw(self, img):
        font = cv2.FONT_HERSHEY_SIMPLEX
        rows = img.shape[0]
        cols = img.shape[1]

        for i in range(len(self.data)):
            cv2.putText(img,self.data[i],(0,  20*(i+1)), font, 0.5,(255,255,255),1,cv2.LINE_AA)
        return img



def draw_circle(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONUP:
        print "(", x, y, ")"
        # cv2.circle(img,(x,y),5,(255,0,0),-1)

def print_corners():
    print "print corners...."
    img = cv2.imread('test_images/snapshots/snapshot_3087.jpg')
    cv2.imshow("print corners", img)
    cv2.setMouseCallback("print corners", draw_circle)
    while(1):
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break
    cv2.destroyAllWindows()
def find_center():
    print "find center:...."
    img = cv2.imread('test_images/snapshots/snapshot_3087.jpg')
    find_center_of_envelop(img)
    cv2.imshow("find_center", img)
    cv2.setMouseCallback("find_center", draw_circle)
    while(1):
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break
    cv2.destroyAllWindows()

def canny():
    import cv2
    import numpy as np
    from matplotlib import pyplot as plt

    img = cv2.imread('road1.jpg')
    # Canny(image, min value, max value, ... aperature size, ...)
    edges = cv2.Canny(img, 100, 200)
    print "edges: ", edges
    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(edges, cmap='gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

    plt.show()

def contours (img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray, (5,5))
    gray = cv2.Canny(gray, 100, 200, apertureSize=3)

    gray = region_of_interest(gray)

    # ret, thresh = cv2.threshold(gray, 127, 255, 0)
    # im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    im2, contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print "here is your contours: len: ", len(contours)
    print "here is hierarchy: ", hierarchy
    # draw ALL the contours! Not the most useful:
    # cv2.drawContours(img, contours, -1, (255,0,0), 3)
    #draw contour [i]:
    i = 2
    for i in range(len(contours)):
        cv2.drawContours(img, contours, i, (255,0,i), 5)
    return img

def region_of_interest(img):
    """
    https://github.com/paramaggarwal/CarND-LaneLines-P1/blob/master/P1.ipynb

    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """

    # row, col, chanels = img.shape
    #number of rows and number of cols in this image
    row = img.shape[0] 
    col = img.shape[1]

    # Let's make vertices a local object:
    vertices = np.array([[(0, row),
                      (col/4, row/2),
                      (col*3/4, row/2),
                      (col, row) ]],
                      dtype=np.int32)



    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def distance_to_line(x, y, line):
    x1, y1, x2, y2 = line
    x_diff = x2 - x1
    y_diff = y2 - y1
    num = abs(y_diff*x - x_diff*y + x2*y1 - y2*x1)
    den = math.sqrt(y_diff**2 + x_diff**2)
    return num / den
def is_in_cluster(clusters, line):
    line_no, x1, y1, x2, y2, length, angle = line
    print "is_in_cluster: clusters: "
    i = 0
    for cluster in clusters:
        if abs(angle - cluster["angle"]/cluster["line_count"])<20: 
            if distance_to_line(x1, y1, cluster["line"])<100:    
                return i    
        i = i + 1
    return -1
def new_cluster(clusters, line):
    cluster = {}
    line_no, x1, y1, x2, y2, length, angle = line
    cluster["max_x"] = max(x1, x2)
    cluster["min_x"] = min(x1, x2)
    cluster["min_y"] = min(y1, y2)
    cluster["max_y"] = max(y1, y2)
    cluster["angle"] = angle
    cluster["avg_angle"] = angle
    cluster["line_count"] = 1
    if (angle <0):
        cluster["x1"] = cluster["min_x"]
        cluster["y1"] = cluster["max_y"]
        cluster["x2"] = cluster["max_x"]
        cluster["y2"] = cluster["min_y"]
    else:
        cluster["x1"] = cluster["min_x"]
        cluster["y1"] = cluster["min_y"]
        cluster["x2"] = cluster["max_x"]
        cluster["y2"] = cluster["max_y"]
    cluster["line"] = [cluster["x1"], cluster["y1"], cluster["x2"], cluster["y2"]]

    clusters.append(cluster)
 
def update_cluster(cluster, line):
    line_no, x1, y1, x2, y2, length, angle = line
    cluster["angle"] = cluster["angle"] + angle
    cluster["line_count"] = cluster["line_count"] + 1
    cluster["avg_angle"] = cluster["angle"]/cluster["line_count"]
    cluster["max_x"] = max(x1, cluster["x1"], cluster["x2"])
    cluster["min_x"] = min(x1, cluster["x1"], cluster["x2"])
    cluster["min_y"] = min(y1, cluster["y1"], cluster["y2"])
    cluster["max_y"] = max(y1, cluster["y1"], cluster["y2"])
    if (angle <0):
        cluster["x1"] = cluster["min_x"]
        cluster["y1"] = cluster["max_y"]
        cluster["x2"] = cluster["max_x"]
        cluster["y2"] = cluster["min_y"]
    else:
        cluster["x1"] = cluster["min_x"]
        cluster["y1"] = cluster["min_y"]
        cluster["x2"] = cluster["max_x"]
        cluster["y2"] = cluster["max_y"]
    cluster["line"] = [cluster["x1"], cluster["y1"], cluster["x2"], cluster["y2"]]

    print "updated cluster: "
    print "   ", cluster

def cluster_lines(info):
    clusters = []
    by_x1 = sorted(info, key = itemgetter(1))
    for line in by_x1:
        line_no, x1, y1, x2, y2, length, angle = line
        # Is this a part of any of the clusters?
        cluster_index = is_in_cluster(clusters, line)
        if cluster_index >=0:
            update_cluster(clusters[cluster_index], line)
        else:
            new_cluster(clusters, line)
    print "-- -- -- -- -- -- -- -- -- -- -- -- -- "
    i = 0
    for cluster in clusters:
        values = cluster.values()
        # print ["{0:.0f}".format(i) for i in values]
        print cluster
    return clusters

def simplifyLines(info):
    by_x1 = sorted(info, key = itemgetter(1))
    last_line = [0,0, 1,1]
    last_angle = 0

    for line in by_x1:
        # print "line: ", line
        line_no, x1, y1, x2, y2, length, angle = line
        d = distance_to_line(x1, y1, last_line)
        angle_diff = abs(angle - last_angle)
        if abs(angle_diff)>20:
            print "~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ "
        if (d>20):
            print ". . . . . . . . . . . . "
        print "distance line: ", line_no, ": ", "{0:0.0f}".format(d), \
            "| this line: ", (x1, y1), ", last line: ", last_line, \
            ", this angle: ", "{0:0.0f}".format(angle), ", last angle: ", "{0:0.0f}".format(last_angle)
        last_line = line[1:5]
        last_angle = angle
def add_cluster_lines(img, clusters):
    for cluster in clusters:
        x1, y1, x2, y2 = cluster["line"]
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 5)
    return img
def hough_linesP(img):
    # hough lines ** probabilistic **:
    # more efficient, gives you the begining and ending of the lines:
    # give me an image object, I will find and draw linesing on it 

    import cv2
    import numpy as np

    text = Text()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray, (5,5))
    edges = cv2.Canny(gray, 100, 200, apertureSize=3)
    edges = region_of_interest(edges)


    #min line length, max line gap?
    lines = cv2.HoughLinesP(edges,rho = 1,theta = 1*np.pi/180,threshold = 100,minLineLength = 100,maxLineGap = 50)
    if lines is None:
        print " . . . . . no lines were found."
        text.add("no lines were found")
        return img , []
    
    print "---------------------"
    #draw all the lines Hough found:
    # print "len(lines): ", len(lines), "lines: "
    # print lines
    # return img
    print " - - - - - - hough_linesP: ", len(lines), " lines found: "
    i = 0 
    info = []
    line_no = 0 
    font = cv2.FONT_HERSHEY_SIMPLEX

    # draw on and return a blank image with lines on it.
    # img = np.zeros(img.shape,  np.uint8)
    for line in lines:
        print ". . . . . . . . . . . . . . ."
        # line is 'normally' a list of one: [[x1, y1, x2, y2]] line[0] has the two points of the line

        x1 = line[0][0]
        y1 = line[0][1]
        x2 = line[0][2]
        y2 = line[0][3]

        line_length = math.sqrt((x2 - x1)*(x2 - x1) + (y2 - y1)*(y2 - y1))
        line_slope = (y2 - y1)/float(x2 - x1)
        line_angle = math.degrees(math.atan(line_slope))
        s = [line_no, x1, y1, x2, y2, line_length, line_angle]
        info.append(s)
        text.add(str(x1)+", "+str(y1)+ "|"+str(x2)+", "+str(y2)+" [len: "+"{0:.2f}".format(line_length)+"| angle: "+"{0:.2f}".format(line_angle)+"]")
        
        print "--------line: ", line, "d: : ", line_length, "slope: ", line_slope, ", angle: ", line_angle
        
        cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 5)
        # cv2.putText(img,str(line_no),(x2,  y2), font, 0.5,(255,255,255),1,cv2.LINE_AA)

        i = i + 50
        line_no = line_no + 1

    i = 0
    for item in info:
        cv2.circle(img, (item[1], item[2]),5,(255,255,0),-1)
        cv2.circle(img, (item[3], item[4]),5,(255,0,0),-1)
        cv2.putText(img,str(item[0]),(item[1]+10,  item[2]), font, 0.5,(0,0,255),1,cv2.LINE_AA)
        cv2.putText(img,str(item[0]),(item[3]+10,  item[4]), font, 0.5,(255,0,0),1,cv2.LINE_AA)

        i = i + 30

        
    by_x1 = sorted(info, key = itemgetter(1))
    by_y1 = sorted(info, key = itemgetter(2))
    by_length = sorted(info, key = itemgetter(5))
    by_angle = sorted(info, key = itemgetter(6))
    print "------- by x1: ----"
    for item in by_x1:
        print ["{0:.0f}".format(i) for i in item]
    print "------- by y1: ----"
    for item in by_y1:
        print ["{0:.0f}".format(i) for i in item]
    print "------- by angle: ----"
    for item in by_angle:
        print ["{0:.0f}".format(i) for i in item]
    print "------- by length: ----"
    for item in by_length:
        print ["{0:.0f}".format(i) for i in item]
    print "------- x1, x2: -------"
    for item in by_x1:
        print "{0:.0f}".format(item[0]) + "| x1: "+ "{0:.0f}".format(item[1]) + ", x2: "+ "{0:.0f}".format(item[3])

    print "------- y1, y2: -------"
    for item in by_y1:
        print "{0:.0f}".format(item[0]) + "| y1: "+ "{0:.0f}".format(item[2]) + ", y2: "+ "{0:.0f}".format(item[4])

    # img = text.draw(img)
    return img, info
def blob_detector_params():
    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()
    
    # filter by color (does not work?)
    params.filterByColor = True
    params.blobColor = 230

    # Change thresholds
    params.minThreshold = 100;
    params.maxThreshold = 2000;
     
    # Filter by Area.
    params.filterByArea = False
    params.minArea = 3000
     
    # Filter by Circularity
    params.filterByCircularity = False
    params.minCircularity = 0.1
     
    # Filter by Convexity
    params.filterByConvexity = False
    params.minConvexity = 0.87
     
    # Filter by Inertia
    params.filterByInertia = False
    params.minInertiaRatio = 0.01
     
    # Create a detector with the parameters
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3 :
        detector = cv2.SimpleBlobDetector(params)
    else : 
        detector = cv2.SimpleBlobDetector_create(params)
    return detector
def detect_blobs(img):
    gray = img
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    # gray = cv2.blur(gray, (5,5))
    # edges = cv2.Canny(gray, 100, 200, apertureSize=3)
    edges = region_of_interest(gray)
    # Set up the detector with default parameters.

    # detector = cv2.SimpleBlobDetector_create()
    detector = blob_detector_params()     
    return_img = edges
    # Detect blobs.
    keypoints = detector.detect(return_img)
     
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(return_img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return im_with_keypoints

def hough_lines(img):
    # give me an image object, I will find and draw lines on it 



    # gray = region_of_interest(img)
    gray = img
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray, (5,5))
    edges = cv2.Canny(gray, 100, 200, apertureSize=3)
    edges = region_of_interest(edges)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # HoughLines args: 
    # image, 
    # ro (1) and 
    # theta accuracy(np.pi/180), 
    # fourth: threashold (200)
    
    #min line length, max line gap?
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    text = Text()

    if lines is None:
        # print " . . . . . no lines were found."
        text.add("No Lines Found.")
        img = text.draw(img)
        return img 
    
    print "---------------------"
    print "len(lines): ", len(lines)
    #draw all the lines Hough found:
    
    # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
    #Show (and draw lines on) only the Canny image:
    # img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

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
            text.add("rho: "+ str(rho)+", theta: " + str(theta)+ " |x1: "+str(x1)+", y1: "+str(y1)+", x2: "+str(x2)+", y2: "+str(y2))
            # draw the lines on the image: 
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

            #. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
            # draw only the first line (and exit.)
            #return img
            #. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
    img = text.draw(img)
    return img

def hough(img):
    import cv2
    import numpy as np

    # img = cv2.imread('road1.jpg')
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


    return img
# birds eye:
# https://wroscoe.github.io/compound-eye-autopilot.html
def four_point_transform(img, pts):

    # maxWidth, maxHeight = img.shape[0], img.shape[1] #300, 300
    print "image shape: ", img.shape
    # maxWidth, maxHeight = 300, 300
    maxWidth, maxHeight = img.shape[:2]
    # hwratio = 11/8.5 #letter size paper
    hwratio = 5.0/4.0 # Paula's envelop: 5" X 4"
    # scale = int(maxWidth/12)
    scale = int(maxWidth/24)
    
    # center_x = 150
    # center_y = 250
    center_x = 556
    center_y = 504
    
    dst = np.array([
    [center_x - scale, center_y - scale*hwratio], #top left
    [center_x + scale, center_y - scale*hwratio], #top right
    [center_x + scale, center_y + scale*hwratio], #bottom right
    [center_x - scale, center_y + scale*hwratio], #bottom left
    ], dtype = "float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(pts, dst)
    
    return M
def find_center_of_envelop(img):
    tl =  ( 451, 445 )
    tr =  ( 666, 434 )
    br =  ( 691, 584 )
    bl =  ( 406, 604 )

    #coordinates of corners: order is [top left, top right, bottom right, bottom left]
    corners = np.array([tl, tr, br, bl], dtype="float32")
    img2 = img.copy()
    for c in corners:
        cv2.circle(img2, tuple(c), 3, (0,255,0), -1)
    cv2.line(img, tl, br, (255,0,0), 1)
    cv2.line(img, tr, bl, (255,0,0), 1)
def birds_eye(img):
    # print corners.... Paula's envelop
    tl =  ( 451, 445 )
    tr =  ( 666, 434 )
    br =  ( 691, 584 )
    bl =  ( 406, 604 )

    #coordinates of corners: order is [top left, top right, bottom right, bottom left]
    corners = np.array([tl, tr, br, bl], dtype="float32")
    img2 = img.copy()
    for c in corners:
        cv2.circle(img2, tuple(c), 3, (0,255,0), -1)

    M = four_point_transform(img2, corners)
    # warped =  cv2.warpPerspective(img2, M, (300, 300))
    warped =  cv2.warpPerspective(img2, M, (img2.shape[1], img2.shape[0]))
    
    return warped


def test_line_one_video():
    cap = cv2.VideoCapture(0)

    #sometimes cap is not opened (?) use cap.open() to open it
    if not cap.isOpened():
        cap.open()


    frame_number = 0
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

        # gray = birds_eye(frame)
        # gray, info = hough_linesP(birds_eye(frame))
        gray, info = hough_linesP(frame)

        cv2.setMouseCallback('frame',draw_circle, gray)
        # Display the resulting frame
        cv2.imshow('frame',gray)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        elif key & 0XFF == ord('1'):
            ret = cap.set(3, cap.get(3)/2)
            ret = cap.set(4, cap.get(4)/2)
        elif key & 0XFF == ord('s'):
            print "   saving frame ... ", frame_number
            cv2.imwrite('test_images/snapshots/' + "snapshot_" + str(frame_number) + ".jpg", frame)

        
        frame_number = frame_number + 1

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
        # - - - - - - - - - - - - - - - - - - - - - -  -
        result, info = hough_linesP(image)
        # run it a second time: 
        # result = hough_linesP(result)
        # simplifyLines(info)
        clusters = cluster_lines(info)
        result = add_cluster_lines(image, clusters) #we do not want the hough lines
        # perspective transformation:
        # result = birds_eye(result)
        
        # result = image
        # result = birds_eye(result)
        # result = detect_blobs(image)
        # result = contours(result)
        # - - - - - - - - - - - - - - - - - - - - - -  -
        # plt.subplot(2, 3, i + 1)
        # plt.imshow(result)
        # mpimg.imsave('test_images/marked/' + image_path[12:-4] + '_detected.jpg', result)
        print 'test_images/marked/' + image_path[len(source_path):-4]
        cv2.imwrite('test_images/marked/' + image_path[len(source_path):-4] + '_detected.jpg', result)
    print ". . . . . . . . . . . . . . . . . . . . ."
    print "Batch processed. "

# canny()
# hough(img)

def main():
    print "------------------------------------------------"
    print "number of command line args: ", len(sys.argv)
    print "------------------------------------------------"
    print ""
    if len(sys.argv) >0:
        print "here is the commandline arguments: "
        # argv[0] is the name of this file
        # the rest are the command line arguments:
        for i in range(len(sys.argv)):
            print  sys.argv[i]
        if sys.argv[1] == "v":
            test_line_one_video()
        elif sys.argv[1] == 'i':
            test_on_images()
        elif sys.argv[1] == "corners":
            print_corners()
        print "------------------------------------------------" 
    print ""
    print ""
    # test_line_one_video()
    # test_on_images()
    # print_corners()
    # find_center()
if __name__ == '__main__':
    main()
