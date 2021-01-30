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
    retMask, coord = myFilter.isolate(hsv=hsv, frame=frame)

    # print(coord)

    # Apply mask to original frame
    processedImg = cv.bitwise_and(frame, frame, mask=retMask)

    # Get the contours in the image
    contourArr = myFinder.detectContours(processedImg)

    # Get contour Areas
    for cont in contourArr:
        area = cv.contourArea(cont)
        # Outline the contours
        if area > 5500:
            frame = cv.drawContours(frame, [cont], -1, (0, 255, 0), 2)

    cv.imshow("Result", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        vidCap.release()
        cv.destroyAllWindows()
        break
