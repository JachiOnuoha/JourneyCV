# Practice using match template

import os
import cv2 as cv
import numpy as np


# Access the file in this current directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def template_match(sourceImg_path, templateImg_path, setThreshold=0.5, matchMethod=cv.TM_CCOEFF_NORMED, showResult=False):

    # Load up the source image and image to be searched
    try:
        templateImg = cv.imread(templateImg_path, cv.IMREAD_UNCHANGED)
        sourceImg = cv.imread(sourceImg_path, cv.IMREAD_UNCHANGED)
    except ValueError:
        print("{} and/or {} not found".format(sourceImg_path, templateImg_path))

    # Get the width and length of the source image and template
    template_w = templateImg.shape[1]
    template_h = templateImg.shape[0]

    # Apply match template to the images
    result = cv.matchTemplate(sourceImg, templateImg, matchMethod)

    # Get the best and worst maatches and their coordinates(top left corners)
    minval, maxval, minLoc, maxLoc = cv.minMaxLoc(result)

    # Set trhreshold for what is considered a good match
    threshold = setThreshold

    # Save coordinates that meet threshold requirement
    locations = np.where(result >= threshold)
    # Reverse and combine to (x,y) format
    locations = list(zip(*locations[::-1]))

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

    if showResult:
        if len(rectangles):

            # Iterate over rectangle elements and create coordinates and dimensions
            for (x, y, w, h) in rectangles:
                coord = (x, y)
                dimensions = (x+w, y+h)

                # Draw a green rectangle on all the matching coordinates
                cv.rectangle(sourceImg, coord, dimensions, (0, 255, 0))

            # Display the source image with the best match identified
            cv.imshow("Results", sourceImg)

            # Close the window when any key is pressed
            cv.waitKey()


# # Uncomment to function below to test
# template_match("farm2.jpeg", "house2.jpeg", setThreshold=0.40, showResult=True)
