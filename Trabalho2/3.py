import cv2
import numpy as np


def fill_holes(image):
    contours, _ = cv2.findContours(
        image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(image)
    for contour in contours:
        cv2.drawContours(filled, [contour], 0, 1, -1)
    return filled


img = cv2.imread('img_cells.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imshow('Original Image', img)
cv2.waitKey(0)

img_blur = cv2.GaussianBlur(img, (5, 5), 0)
cv2.imshow('Blurred Image', img_blur)
cv2.imwrite('output/3/blurred.jpg', img_blur)
cv2.waitKey(0)

_, thresh = cv2.threshold(
    img_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
cv2.imshow('binary Image', thresh)
cv2.imwrite('output/3/binary_image.jpg', thresh)
cv2.waitKey(0)

filled = fill_holes(thresh).astype(np.uint8) * 255
cv2.imshow('Filled Image', filled)
cv2.imwrite('output/3/filled_image.jpg', filled)
cv2.waitKey(0)

kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(filled, cv2.MORPH_OPEN, kernel, iterations=2)
cv2.imshow('Opened Image', opening)
cv2.imwrite('output/3/opened_image.jpg', opening)
cv2.waitKey(0)

sure_bg = cv2.dilate(opening, kernel, iterations=2)
cv2.imshow('Sure Background', sure_bg)
cv2.imwrite('output/3/sure_background.jpg', sure_bg)
cv2.waitKey(0)

dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
cv2.imshow('Sure Foreground', sure_fg)
cv2.imwrite('output/3/sure_foreground.jpg', sure_fg)
cv2.waitKey(0)

sure_fg = np.uint8(sure_fg)
closing = cv2.morphologyEx(sure_fg, cv2.MORPH_CLOSE, kernel, iterations=3)
cv2.imshow('Closed Image', closing)
cv2.imwrite('output/3/closed_image.jpg', closing)
cv2.waitKey(0)

unknown = cv2.subtract(sure_bg, closing)
cv2.imshow('Unknown Regions', unknown)
cv2.imwrite('output/3/unknown_regions.jpg', unknown)
cv2.waitKey(0)

_, markers = cv2.connectedComponents(closing)

markers = markers + 1

markers[unknown == 255] = 0

img_seg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

markers = cv2.watershed(img_seg, markers)

img_seg[markers == -1] = [0, 0, 255]

cv2.imshow('Segmented Image', img_seg)
cv2.imwrite('output/3/segmented_image.jpg', img_seg)
cv2.waitKey(0)
cv2.destroyAllWindows()
