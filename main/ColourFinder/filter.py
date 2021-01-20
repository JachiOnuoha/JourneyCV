import cv2 as cv
import numpy as np


class HSVFilter:
    intensity = 255

    def __init__(self, intensity):
        self.intensity = intensity

    def isolate(self, hsv, frame, colorType="blue"):
        # Gaussian Blur
        hsv = cv.GaussianBlur(hsv, (5, 5), 0)

        # Generate blue color mask
        if colorType == "blue":
            lowerBlue = np.array([110, 50, 50])
            upperBlue = np.array([130, 255, 255])
            mask = cv.inRange(hsv, lowerBlue, upperBlue)

        # Generate red color mask
        elif colorType == "red":
            lowerRed = np.array([])
            upperRed = np.array([])
            mask = cv.inRange(hsv, lowerRed, upperRed)

        # Generate green color mask
        elif colorType == "green":
            lowerGreen = np.array([])
            upperGreen = np.array([])
            mask = cv.inRange(hsv, lowerGreen, upperGreen)

        # Generate yellow color mask
        elif colorType == "yellow":
            lowerYellow = np.array([])
            upperYellow = np.array([])
            mask = cv.inRange(hsv, lowerYellow, upperYellow)

        # Isolate where the color appears and return the coordinates
        result = np.where(mask == self.intensity)
        return mask, result
