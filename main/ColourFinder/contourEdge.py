import cv2 as cv


# Edge and Contour detection class
class EdgeFinder:
    def __init__(self):
        pass

    # Get the contours in an image
    def detectContours(self, img):

        # Threshold the gaussian blurred masked image using otsu binarization
        ret, thresh_img = cv.threshold(
            img, 0, 255, cv.THRESH_BINARY)

        # Get the contuors in the image from the preprocessed image
        contours, hierarchy = cv.findContours(
            thresh_img, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

        return contours

    # Write the color
    @staticmethod
    def colorAnnotation(contourArr, videoFrame, colorType):
        # Get contour Areas
        for cont in contourArr:
            area = cv.contourArea(cont)

            # Outline the contours
            if area > 1000:
                videoFrame = cv.drawContours(
                    videoFrame, [cont], -1, (0, 255, 0), 2)

                # Find the center of mass of the shape for annotation
                M = cv.moments(cont)
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])

                # Draw a white dot and label the color of the object
                cv.circle(videoFrame, (cx, cy), 3, (255, 255, 255), -1)
                cv.putText(videoFrame, colorType, (cx-20, cy-20),
                           cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
