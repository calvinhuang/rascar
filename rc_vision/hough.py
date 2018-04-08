import cv2
import numpy as np


def hough_lines_p(img):
    min_line_length = 0
    max_line_gap = 10

    lines = cv2.HoughLinesP(img, 1, np.pi/180, 20, min_line_length, max_line_gap)

    return lines


def canny(img):
    return cv2.Canny(img, 100, 200, L2gradient=True)


def gauss(img, kernel):
    return cv2.GaussianBlur(img, (kernel, kernel), 0)


def draw_lines(img, lines, dest=None):
    if dest is None:
        dest = img

    if lines is None:
        print("No lines")
        return dest

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(dest, (x1, y1), (x2, y2), thickness=7, color=(0, 255, 0))

    return dest


def roi(img):
    mask = np.zeros_like(img)

    pts = np.array([[(835, 330), (375, 330), (0, 720), (1280, 720)]], dtype=np.int32)

    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, pts, ignore_mask_color)

    masked_image = cv2.bitwise_and(img, mask)

    return masked_image


def process_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    height, width, ch = img.shape

    blur = gauss(gray, 5)

    edges = canny(blur)

    interest = roi(edges)

    lines = hough_lines_p(interest)

    left, right = determine_lanes(lines)

    m, b = determine_slope_intercept(left)

    n, c = determine_slope_intercept(right)

    if check_lanes(m, n):
        p1, p2 = determine_points(m, b, 270, 719)
        p3, p4 = determine_points(n, c, 270, 719)
        x, y = intersecting_lane_point(m, b, n, c)
        vanish_point = draw_vanishing_point(img, x, y)
        processed = draw_lane(vanish_point, p1, p2, p4, p3)
    else:
        print("THROW IMAGE")
        processed = img

    # print((determine_midway(p1, p4, width/2)))

    return processed


def determine_lanes(lines):
    left_lane = []
    right_lane = []

    if lines is None:
        return left_lane, right_lane

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2 or y1 == y2:
                continue
            slope = (y2-y1)/(x2-x1)
            intercept = y1 - slope * x1

            if slope < -1:
                left_lane.append([slope, intercept])
            if slope > 1:
                right_lane.append([slope, intercept])

    return left_lane, right_lane


def determine_slope_intercept(lines):
    total_slope = 0
    total_intercept = 0

    for slope, intercept in lines:
        total_slope += slope
        total_intercept += intercept

    if len(lines) > 0:
        avg_slope = total_slope / len(lines)
        avg_intercept = total_intercept / len(lines)
    else:
        avg_slope = 0
        avg_intercept = 0

    return avg_slope, avg_intercept


def determine_points(slope, intercept, cutoff=0, height=0):
    if slope is 0:
        x1 = x2 = 0
    else:
        x1 = (cutoff - intercept)/slope
        x2 = (height-intercept)/slope

    return (int(x1), cutoff), (int(x2), height)


def draw_lane(img, left_p1, left_p2, right_p1, right_p2, dest=None):
    if dest is None:
        dest = img

    cv2.line(dest, left_p1, left_p2, thickness=5, color=(255, 0, 0))
    cv2.line(dest, right_p1, right_p2, thickness=5, color=(0, 0, 255))

    return dest


def determine_midway(p1, p2, mid_pic):
    mid_of_lane = (p1[0] + p2[0])/2
    error = round(percent_error(mid_pic, mid_of_lane), 4)

    return error


def percent_error(experimental, actual):
    return ((experimental-actual)/actual) * 100


def check_lanes(left_slope, right_slope):
    return check_left_lane(left_slope) and check_right_lane(right_slope)


def check_left_lane(slope):
    return slope < 0


def check_right_lane(slope):
    return slope > 0


def intersecting_lane_point(left_slope, left_intercept, right_slope, right_intercept):
    x = (right_intercept - left_intercept)/(left_slope - right_slope)
    y = (right_slope * x) + right_intercept

    return int(x), int(y)


def draw_vanishing_point(img, x, y, dest=None):
    if dest is None:
        dest = img

    cv2.circle(dest, (x, y), 1, (0, 255, 0), thickness=10)

    return dest


def determine_angle_to_vp(midpoint, height, x, y):
    p1 = (midpoint, height)
    p2 = (midpoint, y)
    p3 = (x, y)

    opp= length_line(p2, p3)
    hyp = length_line(p1, p3)

    theta = np.arcsin(opp/hyp)

    degree = radians_to_degree(theta)

    return degree


def length_line(pt1, pt2):
    a = (pt2[0]-pt1[0])
    b = (pt2[1]-pt1[1])
    length = np.sqrt((a*a)+(b*b))

    return length


def radians_to_degree(radians):
    return (radians*180)/np.pi


if __name__ == '__main__':
    vid = cv2.VideoCapture(0)

    while vid.isOpened():
        ret, frame = vid.read()

        p = process_image(frame)

        cv2.imshow('p', p)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

