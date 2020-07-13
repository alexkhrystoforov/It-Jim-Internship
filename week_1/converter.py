import cv2
import numpy as np


def bgr2hsv_converter(b, g, r):

    bgr = np.uint8([[[b, g, r]]])
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    print(hsv)


bgr2hsv_converter(1, 10, 32)
