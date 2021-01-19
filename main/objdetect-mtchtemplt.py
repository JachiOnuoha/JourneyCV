# Practice using match template

import os
import cv2 as cv
import numpy as np


# Access the file in this file path
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load up the source image and image to be searched
templateImg = cv.imread("house2.jpeg", cv.IMREAD_UNCHANGED)
sourceImg = cv.imread("farm2.jpeg", cv.IMREAD_UNCHANGED)  # 960 x 540


# Get the width and length of the source image and template
template_w = templateImg.shape[1]
template_h = templateImg.shape[0]

# Apply match template to the images
result = cv.matchTemplate(sourceImg, templateImg, cv.TM_CCOEFF_NORMED)

# Get the best and worst maatches and their coordinates(top left corners)
minval, maxval, minLoc, maxLoc = cv.minMaxLoc(result)

# Set trhreshold for what is considered a good match
threshold = 0.40

# Save coordinates that meet threshold requirement
locations = np.where(result >= threshold)
locations = list(zip(*locations[::-1]))  # Reverse and combine to (x,y) format

# Create rectangles which consist of top left coordnate and the width and height of the template
# i.e (x,y,w,h)
rectangles = []
for loc in locations:
    rect = [int(loc[0]), int(loc[1]), int(template_w), int(template_h)]
    rectangles.append(rect)

print("Rectangles before grouping:\n {}\n".format(rectangles))

# Use groupRectangles to group overlapping rectangles which do not differ by more
# than 0.5 apart and require there to be atleast 1 overlap
rectangles, weights = cv.groupRectangles(rectangles, 1, eps=0.5)
print("Rectangles after grouping:\n {}\n".format(rectangles))


if len(rectangles):

    # Iterate over rectangle elements
    for (x, y, w, h) in rectangles:
        coord = (x, y)
        dimensions = (x+w, y+h)

        # Draw a green rectangle on all the matching coordinates
        cv.rectangle(sourceImg, coord, dimensions, (0, 255, 0))

    # Display the source image with the best match identified
    cv.imshow("Results", sourceImg)
    cv.waitKey()
