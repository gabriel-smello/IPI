import cv2
import numpy as np

car = cv2.imread("inputs/car.png")
crowd = cv2.imread("inputs/crowd.png")
university = cv2.imread("inputs/university.png")

# Valores Gamma
gammas = [0.5, 1.5]

# Realce car
for gamma in gammas:
    realce_car = cv2.pow(car / 255.0, gamma)
    realce_car = cv2.convertScaleAbs(realce_car * 255)

    # Imagem pós realce
    cv2.imshow(f"Car - Gamma = {gamma}", realce_car)

    # Salvar
    cv2.imwrite(f'outputs/car-gamma{gamma}.png', realce_car)

# realce crowd
for gamma in gammas:
    realce_crowd = cv2.pow(crowd / 255.0, gamma)
    realce_crowd = cv2.convertScaleAbs(realce_crowd * 255)

    # Imagem pós realce
    cv2.imshow(f"Crowd - Gamma = {gamma}", realce_crowd)

    # Salvar
    cv2.imwrite(f'outputs/crowd-gamma{gamma}.png', realce_crowd)

# realce university
for gamma in gammas:
    realce_university = cv2.pow(university / 255.0, gamma)
    realce_university = cv2.convertScaleAbs(realce_university * 255)

    # Imagem pós realce
    cv2.imshow(f"University - Gamma = {gamma}", realce_university)

    # Salvar
    cv2.imwrite(f'outputs/university-gamma{gamma}.png', realce_university)

cv2.waitKey(0)
cv2.destroyAllWindows()
