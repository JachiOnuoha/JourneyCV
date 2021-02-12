import cv2 as cv


# Edge and Contour detection class
class EdgeFinder:
    def __init__(self):
        pass

    # def applyCanny(self, maskedFrame=None, minVal=100, maxVal=200, useL2=True):
    #     edges = cv.Canny(maskedFrame, minVal, maxVal, L2gradient=useL2)
    #     return edges

    # Get the contours in an image
    def detectContours(self, img):
        # Convert the image to from HSV to grayscale
        img = cv.cvtColor(img, cv.COLOR_HSV2BGR)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Threshold the gaussian blurred masked image using otsu binarization
        ret, thresh_img = cv.threshold(
            img, 0, 255, cv.THRESH_BINARY)

        # # Detect the edges in the image
        # edges = cv.Canny(img, 100, 200, L2gradient=True)

        # Get the contuors in the image from the preprocessed image
        contours, hierarchy = cv.findContours(
            thresh_img, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

        return thresh_img, contours
