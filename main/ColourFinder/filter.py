import cv2 as cv
import numpy as np


class HSVFilter:
    intensity = 255

    def __init__(self, intensity):
        self.intensity = intensity

    def isolate(self, hsv, frame):
        # Gaussian Blur
        # hsv = cv.GaussianBlur(hsv, (5, 5), 0)

        # Generate blue color mask
        lowerBlue = np.array([100, 80, 0])
        upperBlue = np.array([140, 255, 255])
        mask1 = cv.inRange(hsv, lowerBlue, upperBlue)

        # Generate red color mask
        lowerRed = np.array([0, 50, 120])
        upperRed = np.array([10, 255, 255])
        mask2 = cv.inRange(hsv, lowerRed, upperRed)

        # Generate green color mask
        lowerGreen = np.array([45, 70, 80])
        upperGreen = np.array([70, 255, 255])
        mask3 = cv.inRange(hsv, lowerGreen, upperGreen)

        # Generate yellow color mask
        lowerYellow = np.array([25, 70, 120])
        upperYellow = np.array([30, 255, 255])
        mask4 = cv.inRange(hsv, lowerYellow, upperYellow)

        all_masks = [mask1, mask2, mask3, mask4]

        # Isolate where the color appears and return the coordinates
        return all_masks
