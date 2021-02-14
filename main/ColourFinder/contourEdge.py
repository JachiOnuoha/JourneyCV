import cv2 as cv


# Edge and Contour detection class
class EdgeFinder:
    def __init__(self):
        pass

    # Get the contours in an image
    def detectContours(self, img):
        # Convert the image to from HSV to grayscale
        # img = cv.cvtColor(img, cv.COLOR_HSV2BGR)
        # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Threshold the gaussian blurred masked image using otsu binarization
        ret, thresh_img = cv.threshold(
            img, 0, 255, cv.THRESH_BINARY)

        # Get the contuors in the image from the preprocessed image
        contours, hierarchy = cv.findContours(
            thresh_img, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

        return thresh_img, contours
