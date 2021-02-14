# Detect the color of an object by applying an HSV filter to the image

import cv2 as cv
from filter import HSVFilter
from contourEdge import EdgeFinder

# Capture video from webcam
vidCap = cv.VideoCapture(0)

# Create HSV object for filtration
myFilter = HSVFilter(255)
myFinder = EdgeFinder()


while True:
    # Read each frame and ignore the returned value for the end of frame since live videos have infinte frames
    _, frame = vidCap.read()

    # Convert the frame to hsv for filter processing
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Get resulting mask and coordinates for desired color
    blue_mask, red_mask, green_mask, yellow_mask = myFilter.isolate(hsv=hsv, frame=frame)

    # Get the contours in the image
    blueContour = myFinder.detectContours(blue_mask)
    redContour = myFinder.detectContours(red_mask)
    greenContour = myFinder.detectContours(green_mask)
    yellowContour = myFinder.detectContours(yellow_mask)

    # Label the colors
    processFrame = frame
    EdgeFinder.colorAnnotation(blueContour, processFrame, "blue")
    EdgeFinder.colorAnnotation(redContour, processFrame, "red")

    cv.imshow("Result", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        vidCap.release()
        cv.destroyAllWindows()
        break
