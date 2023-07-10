import cv2
import numpy as np

page_image = cv2.imread('morf_test.png', cv2.IMREAD_GRAYSCALE)

_, binary_image = cv2.threshold(
    page_image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow("Binarização otsu", binary_image)
cv2.imwrite('output/2/binary.png', binary_image)

selem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
background = cv2.morphologyEx(page_image, cv2.MORPH_CLOSE, selem)
cv2.imshow("Fechamento", background)
cv2.imwrite('output/2/backgound.png', background)

background_removed = background - page_image
cv2.imshow("background removido", background_removed)
cv2.imwrite('output/2/background_removed.png', background_removed)

bhat = cv2.morphologyEx(page_image, cv2.MORPH_BLACKHAT, selem)
cv2.imshow("blackhat", bhat)
cv2.imwrite('output/2/blackhat.png', bhat)

_, bhat_thresh = cv2.threshold(
    bhat, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow("binary blackhat", bhat_thresh)
cv2.imwrite('output/2/blackhat_binary.png', bhat_thresh)

_, labels, stats, _ = cv2.connectedComponentsWithStats(
    bhat_thresh.astype(np.uint8), connectivity=8)
remove_pequeno = np.zeros_like(labels)
for label, stat in enumerate(stats[1:], start=1):
    if stat[cv2.CC_STAT_AREA] >= 5:
        remove_pequeno[labels == label] = 255
remove_pequeno = remove_pequeno.astype(bool)
remove_pequeno = remove_pequeno.astype(np.uint8) * 255

cv2.imshow("remoção detalhes", remove_pequeno)
cv2.imwrite('output/2/remove_detalhes.png', remove_pequeno)


cv2.imshow("imagem final", cv2.bitwise_not(remove_pequeno))
cv2.imwrite('output/2/imagem_final.png', cv2.bitwise_not(remove_pequeno))

cv2.waitKey(0)
cv2.destroyAllWindows()
