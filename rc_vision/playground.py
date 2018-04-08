import cv2
import matplotlib.pyplot as plt
import numpy as np

while(1):
    img = cv2.imread('test.jpg')
    cv2.imshow('i', img)