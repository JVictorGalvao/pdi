import cv2
import numpy as np
from matplotlib import pyplot as plt


def criarFigura(img, label):
    plt.figure(label)
    plt.imshow(img)
    plt.xticks([]), plt.yticks([])


def imprimeCanaisRGB(R, G, B, shape):
    # Transforma os canais em arrays de 3 posições com as outras duas zeradas
    zeros = np.zeros(shape, dtype="uint8")
    r_zero = np.dstack((R, zeros, zeros))
    g_zero = np.dstack((zeros, G, zeros))
    b_zero = np.dstack((zeros, zeros, B))
    # Junta os canais em uma imagem RGB novamente
    rgb = np.dstack((R, G, B))

    criarFigura(b_zero, 'B')
    criarFigura(g_zero, 'G')
    criarFigura(r_zero, 'R')
    criarFigura(rgb, 'Rgb2')


def main():
    # Abre a imagem
    img = cv2.imread('testpat.1k.color2.tif')
    # Separa os canais RGB
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    criarFigura(img, 'RGB')
    imprimeCanaisRGB(R, G, B, img.shape[:2])
    plt.show()


main()
