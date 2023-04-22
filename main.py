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


def rgbParaYiq(R, G, B):
    r_norm = R/255
    g_norm = G/255
    b_norm = B/255

    y = 0.299*r_norm + 0.587*g_norm + 0.114*b_norm
    i = 0.596*r_norm - 0.274*g_norm - 0.322*b_norm
    q = 0.211*r_norm - 0.523*g_norm + 0.312*b_norm

    return y, i, q


def yiqParaRgb(Y, I, Q):
    r = Y + 0.956*I + 0.621*Q
    g = Y - 0.272*I - 0.647*Q
    b = Y - 1.106*I + 1.703*Q

    r_norm = np.uint8(np.clip(r*255, 0, 255))
    g_norm = np.uint8(np.clip(g*255, 0, 255))
    b_norm = np.uint8(np.clip(b*255, 0, 255))

    return r_norm, g_norm, b_norm


def converteRgbparaYiqParaRgb(R, G, B):
    # converte para yiq
    y, i, q = rgbParaYiq(R, G, B)
    yiq = np.dstack((y, i, q))
    yiq = np.clip(yiq, 0, 1)
    # converte para rgb
    r, g, b = yiqParaRgb(y, i, q)
    rgb = np.dstack((r, g, b))
    rgb = np.clip(rgb, 0, 255)
    # cria figuras
    criarFigura(yiq, 'YIQ')
    criarFigura(rgb, 'RGB2')


def main():
    # Abre a imagem
    img = cv2.imread('testpat.1k.color2.tif')
    # Separa os canais RGB
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    criarFigura(img, 'RGB')
    # imprimeCanaisRGB(R, G, B, img.shape[:2])
    converteRgbparaYiqParaRgb(R, G, B)

    plt.show()


main()
