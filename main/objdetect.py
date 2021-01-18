import os
import cv2 as cv
import numpy as np


# Access the fle in this file path
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load up the source image and image to be searched
templateImg = cv.imread("Icarus2.jpeg", cv.IMREAD_UNCHANGED)
sourceImg = cv.imread("Icarus_and_Sunia.jpeg", cv.IMREAD_UNCHANGED)

result = cv.matchTemplate(sourceImg, templateImg, cv.TM_CCOEFF_NORMED)
minval, maxval, minLoc, maxLoc = cv.minMaxLoc(result)

print(maxLoc)
print(maxval)

template_w = templateImg.shape[1]
template_h = templateImg.shape[0]

bottom = (maxLoc[0]+template_w, maxLoc[1]+template_h)

# cv.imshow("Result", result)
# cv.waitKey()
# cv.imshow("Result", sourceImg)
# cv.waitKey()
cv.rectangle(sourceImg, maxLoc, bottom, (0, 255, 0))
cv.imshow("Results", sourceImg)
cv.waitKey()
# print(maxLoc)
