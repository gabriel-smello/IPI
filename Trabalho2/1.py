import numpy as np
import cv2


def fill_holes(image):
    contours, _ = cv2.findContours(
        image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(image)
    for contour in contours:
        cv2.drawContours(filled, [contour], 0, 1, -1)
    return filled


pcb_image = cv2.imread('pcb.jpg', cv2.IMREAD_GRAYSCALE)
pcb_image = (pcb_image >= 128).astype(np.uint8)*255

cv2.imshow('imagem binaria', pcb_image)
cv2.imwrite('output/1/pcb_orig.jpg', pcb_image)

pcb_filled = fill_holes(pcb_image).astype(np.uint8)*255

cv2.imshow('imagem preenchida', pcb_filled)
cv2.imwrite('output/1/pcb_filled.jpg', pcb_filled)

pcb_holes = cv2.subtract(pcb_filled,  pcb_image)

cv2.imshow('buracos', pcb_holes)
cv2.imwrite('output/1/pcb_holes.jpg', pcb_holes)

_, _, stats, _ = cv2.connectedComponentsWithStats(
    pcb_holes)
area = stats[1:, 4]

for obj_ind, obj_area in enumerate(area):
    diameter = 2 * np.sqrt(obj_area / np.pi)

cv2.waitKey(0)
cv2.destroyAllWindows()

print("Número de buracos:", len(area))
for obj_ind, obj_area in enumerate(area):
    diameter = 2 * np.sqrt(obj_area / np.pi)  # Assuming circular objects
    print("Buraco", obj_ind + 1, "- Diâmetro:", diameter, "px")
