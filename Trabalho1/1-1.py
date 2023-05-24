import numpy as np
import cv2


def TAM2(image, factor):
    height, width, _ = image.shape

    new_height = height * factor
    new_width = width * factor

    result_image = cv2.resize(
        image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

    new_rows = np.repeat(result_image[-1:, :], factor, axis=0)
    result_image = np.vstack((result_image, new_rows))

    new_cols = np.repeat(result_image[:, -1:], factor, axis=1)
    result_image = np.hstack((result_image, new_cols))

    return result_image

###################### APLICAÇÃO #######################


image = cv2.imread('inputs/fruit1.jpg')

new_image2 = TAM2(image, 2)
new_image8 = TAM2(image, 8)


# imagem original
cv2.imshow('Imagem Original', image)

# imagens redimensionada
cv2.imshow('Imagem Redimensionada 2', new_image2)
cv2.imshow('Imagem Redimensionada 8', new_image8)

# Salvar
cv2.imwrite('outputs/fruit1-1-2.jpg', new_image2)
cv2.imwrite('outputs/fruit1-1-8.jpg', new_image8)


cv2.waitKey(0)
cv2.destroyAllWindows()
