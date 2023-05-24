import numpy as np
import cv2


def TAMM(image):
    # Obtém a altura e a largura da imagem original
    height, width, _ = image.shape

    # Calcula a nova altura e largura
    new_height = height * 2
    new_width = width * 2

    # Cria uma matriz vazia para a imagem redimensionada
    resized_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    # Preenche os pixels nas filas que não estão totalmente "vazias"
    for i in range(height):
        for j in range(width):
            resized_image[i*2, j*2] = image[i, j]

            if i < height - 1:
                resized_image[i*2 + 1, j*2] = np.ceil(
                    np.mean([image[i, j], image[i+1, j]], axis=0)).astype(np.uint8)

            if j < width - 1:
                resized_image[i*2, j*2 + 1] = np.ceil(
                    np.mean([image[i, j], image[i, j+1]], axis=0)).astype(np.uint8)

            if i < height - 1 and j < width - 1:
                resized_image[i*2 + 1, j*2 + 1] = np.ceil(np.mean(
                    [image[i, j], image[i+1, j], image[i, j+1], image[i+1, j+1]], axis=0)).astype(np.uint8)

    for j in range(width):
        resized_image[new_height-1, j*2] = resized_image[new_height-2, j*2]
        resized_image[new_height-1, j*2 + 1] = resized_image[new_height-1, j*2]

    for i in range(new_height):
        resized_image[i, new_width-1] = resized_image[i, new_width-2]
    return resized_image


# Exemplo de uso
image = cv2.imread('inputs/fruit1.jpg')

# Amplia a imagem utilizando a função TAMM
new_image = TAMM(image)
cv2.imwrite('outputs/fruit1-2.jpg', new_image)

# Mostra a imagem original
cv2.imshow('Imagem Original', image)

# Mostra a imagem ampliada
cv2.imshow('Imagem Ampliada', new_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
