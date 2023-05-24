import numpy as np
import cv2


def SUPERRES(image1, image2):
    # Verifica se as duas imagens têm as mesmas dimensões
    if image1.shape != image2.shape:
        raise ValueError("As duas imagens devem ter as mesmas dimensões.")

    # Obtém a altura e a largura das imagens originais
    height, width, _ = image1.shape

    # Calcula a nova altura e largura para a imagem final
    new_height = height * 2
    new_width = width * 2

    # Cria uma matriz vazia para a imagem final
    superres_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    # Combina as duas imagens para gerar a imagem final
    for i in range(height):
        for j in range(width):
            # Posições ímpares (image1)
            superres_image[i*2 + 1, j*2 + 1] = image1[i, j]

            # Posições pares (image2)
            superres_image[i*2, j*2] = image2[i, j]

    return superres_image


# Carrega as duas imagens
image1 = cv2.imread('inputs/fruit1.jpg')
image2 = cv2.imread('inputs/fruit2.jpg')

# Chama a função SUPERRES
superres_image = SUPERRES(image1, image2)
cv2.imwrite('outputs/fruit1-3.jpg', superres_image)

# Mostra a primeira imagem
cv2.imshow('Primeira Imagem', image1)

# Mostra a segunda imagem
cv2.imshow('Segunda Imagem', image2)

# Mostra a imagem super-resolvida
cv2.imshow('Imagem Super-resolvida', superres_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
