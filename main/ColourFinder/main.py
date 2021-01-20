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
    ret, coord = myFilter.isolate(hsv=hsv, frame=frame)

    # print(coord)

    # Apply mask to original frame
    final = cv.bitwise_and(frame, frame, mask=ret)

    # Get the contours in the image
    contourArr = myFinder.detectContours(final)

    # Outline the contours
    final = cv.drawContours(frame, contourArr, -1, (0, 255, 0), 2)
    cv.imshow("Result", final)

    if cv.waitKey(1) & 0xFF == ord('q'):
        vidCap.release()
        cv.destroyAllWindows()
        break
