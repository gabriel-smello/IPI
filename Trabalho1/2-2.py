import cv2
import numpy as np
import matplotlib.pyplot as plt

# Show histograma e CDF


def plot_histogram_cdf(image, title):
    hist = cv2.calcHist([image], [0], None, [256], [0, 255])
    cdf = hist.cumsum()

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.bar(np.arange(256), hist.ravel(), color='b')
    plt.title('Histograma ' + title)
    plt.xlabel('Intensidade')
    plt.ylabel('Frequência')

    plt.subplot(1, 2, 2)
    plt.plot(cdf, color='r')
    plt.title('CDF ' + title)
    plt.xlabel('Intensidade')
    plt.ylabel('CDF')

    plt.tight_layout()
    plt.show()


car = cv2.imread('inputs/car.png', 0)
crowd = cv2.imread('inputs/crowd.png', 0)
university = cv2.imread('inputs/university.png', 0)

# Equalização
car_equalized = cv2.equalizeHist(car)
crowd_equalized = cv2.equalizeHist(crowd)
university_equalized = cv2.equalizeHist(university)

# Mostrar resultados das imagens
cv2.imshow('Car', car_equalized)
cv2.imshow('Crowd', crowd_equalized)
cv2.imshow('University', university_equalized)

# Salvar
cv2.imwrite('outputs/car-equalization.png', car_equalized)
cv2.imwrite('outputs/crowd-equalization.png', crowd_equalized)
cv2.imwrite('outputs/university-equalization.png', university_equalized)

# histograma e CDF antes e depois da equalização

    # plot_histogram_cdf(crowd, 'crowd - Antes da equalização')
    # plot_histogram_cdf(crowd_equalized, 'crowd - Após a equalização')

    # plot_histogram_cdf(university, 'university - Antes da equalização')
    # plot_histogram_cdf(university_equalized, 'university - Após a equalização')

plot_histogram_cdf(car, 'Car - Antes da equalização')
plot_histogram_cdf(car_equalized, 'Car - Após a equalização')

cv2.waitKey(0)
cv2.destroyAllWindows()
