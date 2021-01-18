# Practice using match template

import os
import cv2 as cv
import numpy as np


# Access the fle in this file path
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load up the source image and image to be searched
templateImg = cv.imread("Icarus2.jpeg", cv.IMREAD_UNCHANGED)
sourceImg = cv.imread("Icarus_and_Sunia.jpeg", cv.IMREAD_UNCHANGED)

# Get the width and length of the source image and template
template_w = templateImg.shape[1]
template_h = templateImg.shape[0]

# Apply match template to the images
result = cv.matchTemplate(sourceImg, templateImg, cv.TM_CCOEFF_NORMED)

# Get the best and worst maatches and their coordinates(top left corners)
minval, maxval, minLoc, maxLoc = cv.minMaxLoc(result)

# Create the rectangles dimensions
bottom = (maxLoc[0]+template_w, maxLoc[1]+template_h)

# Draw a rectangle on the best match
cv.rectangle(sourceImg, maxLoc, bottom, (0, 255, 0))
cv.imshow("Results", sourceImg)
cv.waitKey()
