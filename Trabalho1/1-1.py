import numpy as np
import cv2


def TAM2(image, factor):
    # Obtém a altura e a largura da imagem original
    height, width, _ = image.shape

    # Calcula a nova altura e largura multiplicando pelo fator
    new_height = height * factor
    new_width = width * factor

    # Redimensiona a imagem para a nova altura e largura
    resized_image = cv2.resize(
        image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

    # Repete a última linha e coluna para preencher as novas linhas e colunas
    repeated_rows = np.repeat(resized_image[-1:, :], factor, axis=0)
    resized_image = np.vstack((resized_image, repeated_rows))

    repeated_cols = np.repeat(resized_image[:, -1:], factor, axis=1)
    resized_image = np.hstack((resized_image, repeated_cols))

    return resized_image


# Carrega a imagem
image = cv2.imread('inputs/fruit1.jpg')

# Chama a função TAM2
new_image2 = TAM2(image, 2)
new_image8 = TAM2(image, 8)


# Mostra a imagem original
cv2.imshow('Imagem Original', image)

# Mostra a imagem redimensionada
cv2.imshow('Imagem Redimensionada 2', new_image2)
cv2.imshow('Imagem Redimensionada 8', new_image8)

# Salva a imagem redimensionada
cv2.imwrite('outputs/fruit1-1-2.jpg', new_image2)
cv2.imwrite('outputs/fruit1-1-8.jpg', new_image8)


cv2.waitKey(0)
cv2.destroyAllWindows()
