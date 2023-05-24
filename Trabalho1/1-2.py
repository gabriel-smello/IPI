import numpy as np
import cv2


def TAMM(image):
    height, width, _ = image.shape

    new_height = height * 2
    new_width = width * 2

    # Matriz vazia
    new_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            new_image[i*2, j*2] = image[i, j]

            if i < height - 1:
                new_image[i*2 + 1, j*2] = np.ceil(
                    np.mean([image[i, j], image[i+1, j]], axis=0)).astype(np.uint8)

            if j < width - 1:
                new_image[i*2, j*2 + 1] = np.ceil(
                    np.mean([image[i, j], image[i, j+1]], axis=0)).astype(np.uint8)

            if i < height - 1 and j < width - 1:
                new_image[i*2 + 1, j*2 + 1] = np.ceil(np.mean(
                    [image[i, j], image[i+1, j], image[i, j+1], image[i+1, j+1]], axis=0)).astype(np.uint8)

    for j in range(width):
        new_image[new_height-1, j*2] = new_image[new_height-2, j*2]
        new_image[new_height-1, j*2 + 1] = new_image[new_height-1, j*2]

    for i in range(new_height):
        new_image[i, new_width-1] = new_image[i, new_width-2]
    return new_image

###################### APLICAÇÃO #######################


image = cv2.imread('inputs/fruit1.jpg')

new_image = TAMM(image)

# imagem original
cv2.imshow('Imagem Original', image)

# imagem redimensionada
cv2.imshow('Imagem Ampliada', new_image)

# Salvar
cv2.imwrite('outputs/fruit1-2.jpg', new_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
