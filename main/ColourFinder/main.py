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
    thresh, contourArr = myFinder.detectContours(processedImg)

    # Get contour Areas
    for cont in contourArr:
        area = cv.contourArea(cont)

        # Outline the contours
        if area > 1000:
            frame = cv.drawContours(frame, [cont], -1, (0, 255, 0), 2)

            # Find the center of mass of the shape for annotation
            M = cv.moments(cont)
            cx = int(M["m10"]/M["m00"])
            cy = int(M["m01"]/M["m00"])

            # Draw a white dot and label the color of the object
            cv.circle(frame, (cx, cy), 3, (255, 255, 255), -1)
            cv.putText(frame, "blue", (cx-20, cy-20), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv.imshow("Result", frame)
    cv.imshow("Threshold", thresh)

    if cv.waitKey(1) & 0xFF == ord('q'):
        vidCap.release()
        cv.destroyAllWindows()
        break
