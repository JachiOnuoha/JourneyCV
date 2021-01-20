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
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        edges = cv.Canny(img, 100, 200, L2gradient=True)

        contours, hierarchy = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

        cnt = contours[4]
        res = cv.drawContours(img, [cnt], 0, (0, 255, 0), 3)
        return res
