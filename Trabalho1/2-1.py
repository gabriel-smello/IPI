import cv2
import numpy as np

# Carregando as imagens
car = cv2.imread("inputs/car.png")
crowd = cv2.imread("inputs/crowd.png")
university = cv2.imread("inputs/university.png")

# Definindo os valores de gamma para o realce
gammas = [0.5, 1.5]

# Realizando o realce power-law para a imagem "car.png"
for gamma in gammas:
    car_power_law = cv2.pow(car / 255.0, gamma)
    car_power_law = cv2.convertScaleAbs(car_power_law * 255)
    cv2.imshow(f"Car - Gamma {gamma}", car_power_law)
    cv2.imwrite(f'outputs/car-gamma{gamma}.png', car_power_law)

# Realizando o realce power-law para a imagem "crowd.png"
for gamma in gammas:
    crowd_power_law = cv2.pow(crowd / 255.0, gamma)
    crowd_power_law = cv2.convertScaleAbs(crowd_power_law * 255)
    cv2.imshow(f"Crowd - Gamma {gamma}", crowd_power_law)
    cv2.imwrite(f'outputs/crowd-gamma{gamma}.png', crowd_power_law)

# Realizando o realce power-law para a imagem "university.png"
for gamma in gammas:
    university_power_law = cv2.pow(university / 255.0, gamma)
    university_power_law = cv2.convertScaleAbs(university_power_law * 255)
    cv2.imshow(f"University - Gamma {gamma}", university_power_law)
    cv2.imwrite(f'outputs/university-gamma{gamma}.png', university_power_law)

cv2.waitKey(0)
cv2.destroyAllWindows()
