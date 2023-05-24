import numpy as np
import cv2


def SUPERRES(image1, image2):
    # Verifica se as duas imagens têm as mesmas dimensões
    if image1.shape != image2.shape:
        raise ValueError("Imagens de dimensões diferentes.")

    height, width, _ = image1.shape

    new_height = height * 2
    new_width = width * 2

    # Matriz vazia
    new_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            # Ímpar (imagem 1)
            new_image[i*2 + 1, j*2 + 1] = image1[i, j]

            # Par (imagem 2)
            new_image[i*2, j*2] = image2[i, j]

    return new_image

###################### APLICAÇÃO #######################


image1 = cv2.imread('inputs/fruit1.jpg')
image2 = cv2.imread('inputs/fruit2.jpg')

new_image = SUPERRES(image1, image2)

# Imagem 1
cv2.imshow('Primeira Imagem', image1)

# Imagem 2
cv2.imshow('Segunda Imagem', image2)

# Imagem após SUPERRES
cv2.imshow('Imagem Super-resolvida', new_image)

# Salvar
cv2.imwrite('outputs/fruit1-3.jpg', new_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
