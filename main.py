import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve2d
import time


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

    criarFigura(b_zero, 'B')
    criarFigura(g_zero, 'G')
    criarFigura(r_zero, 'R')


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


def negativoRgb(img):
    negativo = 255-img
    criarFigura(negativo, 'Negativo RGB')


def negativoEmY(R, G, B):
    y, i, q = rgbParaYiq(R, G, B)
    y = 1-y
    r, g, b = yiqParaRgb(y, i, q)
    negativo_y = np.dstack((r, g, b))
    negativo_y = np.clip(negativo_y, 0, 255)
    criarFigura(negativo_y, 'Negativo em Y')


def filtroSoma(R, G, B):
    # abre o arquivo com filtro
    with open('soma.txt', 'r') as f:
        filtro = np.array([[int(num) for num in line.split()]
                          for line in f])

    canal_r = convolve2d(R, filtro, mode='same', boundary='fill', fillvalue=0)
    canal_g = convolve2d(G, filtro, mode='same', boundary='fill', fillvalue=0)
    canal_b = convolve2d(B, filtro, mode='same', boundary='fill', fillvalue=0)

    img = np.dstack((canal_r, canal_g, canal_b))
    img = np.clip(img, 0, 255)

    criarFigura(img, "SOMA")


def filtroMedia(R, G, B):
    # abre o arquivo com filtro
    with open('media11x11.txt', 'r') as f:
        filtro = np.array([[float(num) for num in line.split()]
                          for line in f])

    canal_r = convolve2d(R, filtro, mode='same', boundary='fill', fillvalue=0)
    canal_g = convolve2d(G, filtro, mode='same', boundary='fill', fillvalue=0)
    canal_b = convolve2d(B, filtro, mode='same', boundary='fill', fillvalue=0)

    img = np.dstack((canal_r, canal_g, canal_b)).astype(np.uint8)

    criarFigura(img, "MEDIA")


def filtroMediaDuplo(R, G, B):
    # abre o arquivo com filtro
    with open('media1x11.txt', 'r') as f:
        filtro = np.array([[float(num) for num in line.split()]
                          for line in f])

    canal_r = convolve2d(R, filtro, mode='same', boundary='fill', fillvalue=0)
    canal_g = convolve2d(G, filtro, mode='same', boundary='fill', fillvalue=0)
    canal_b = convolve2d(B, filtro, mode='same', boundary='fill', fillvalue=0)

    with open('media11x1.txt', 'r') as f:
        filtro = np.array([[float(num) for num in line.split()]
                           for line in f])

    canal_rr = convolve2d(canal_r, filtro, mode='same',
                          boundary='fill', fillvalue=0)
    canal_gg = convolve2d(canal_g, filtro, mode='same',
                          boundary='fill', fillvalue=0)
    canal_bb = convolve2d(canal_b, filtro, mode='same',
                          boundary='fill', fillvalue=0)

    img = np.dstack((canal_rr, canal_gg, canal_bb)).astype(np.uint8)

    criarFigura(img, "MEDIADUPLA")


def filtroSobelH(R, G, B):
    r_min = np.min(R)
    r_max = np.max(R)
    g_min = np.min(G)
    g_max = np.max(G)
    b_min = np.min(B)
    b_max = np.max(B)
    L = 256
    R = ((R - r_min) * ((L-1)/(r_max - r_min))).astype(np.uint8)
    G = ((G - g_min) * ((L-1)/(g_max - g_min))).astype(np.uint8)
    B = ((B - b_min) * ((L-1)/(b_max - b_min))).astype(np.uint8)
    # abre o arquivo com filtro
    with open('sobelh.txt', 'r') as f:
        filtro = np.array([[int(num) for num in line.split()]
                          for line in f])

    canal_r = convolve2d(R, filtro, mode='same', boundary='fill', fillvalue=0)
    canal_g = convolve2d(G, filtro, mode='same', boundary='fill', fillvalue=0)
    canal_b = convolve2d(B, filtro, mode='same', boundary='fill', fillvalue=0)

    img = np.dstack((canal_r, canal_g, canal_b))
    img = np.clip(img, 0, 255)

    criarFigura(img, "SOBELH")


def filtroSobelV(R, G, B):
    r_min = np.min(R)
    r_max = np.max(R)
    g_min = np.min(G)
    g_max = np.max(G)
    b_min = np.min(B)
    b_max = np.max(B)
    L = 256
    R = ((R - r_min) * ((L-1)/(r_max - r_min))).astype(np.uint8)
    G = ((G - g_min) * ((L-1)/(g_max - g_min))).astype(np.uint8)
    B = ((B - b_min) * ((L-1)/(b_max - b_min))).astype(np.uint8)
    # abre o arquivo com filtro
    with open('sobelv.txt', 'r') as f:
        filtro = np.array([[int(num) for num in line.split()]
                          for line in f])

    canal_r = convolve2d(R, filtro, mode='same', boundary='fill', fillvalue=0)
    canal_g = convolve2d(G, filtro, mode='same', boundary='fill', fillvalue=0)
    canal_b = convolve2d(B, filtro, mode='same', boundary='fill', fillvalue=0)

    img = np.dstack((canal_r, canal_g, canal_b))
    img = np.clip(img, 0, 255)

    criarFigura(img, "SOBELV")


def filtroSobel(R, G, B):

    r_min = np.min(R)
    r_max = np.max(R)
    g_min = np.min(G)
    g_max = np.max(G)
    b_min = np.min(B)
    b_max = np.max(B)
    L = 256
    R = ((R - r_min) * ((L-1)/(r_max - r_min))).astype(np.uint8)
    G = ((G - g_min) * ((L-1)/(g_max - g_min))).astype(np.uint8)
    B = ((B - b_min) * ((L-1)/(b_max - b_min))).astype(np.uint8)

    # abre o arquivo com filtro
    with open('sobelh.txt', 'r') as f:
        filtro = np.array([[int(num) for num in line.split()]
                          for line in f])

    canal_r = convolve2d(R, filtro, mode='same', boundary='fill', fillvalue=0)
    canal_g = convolve2d(G, filtro, mode='same', boundary='fill', fillvalue=0)
    canal_b = convolve2d(B, filtro, mode='same', boundary='fill', fillvalue=0)

    with open('sobelv.txt', 'r') as f:
        filtro = np.array([[int(num) for num in line.split()]
                          for line in f])

    canal_rr = convolve2d(canal_r, filtro, mode='same',
                          boundary='fill', fillvalue=0)
    canal_gg = convolve2d(canal_g, filtro, mode='same',
                          boundary='fill', fillvalue=0)
    canal_bb = convolve2d(canal_b, filtro, mode='same',
                          boundary='fill', fillvalue=0)

    img = np.dstack((canal_rr, canal_gg, canal_bb))
    img = np.clip(img, 0, 255)

    criarFigura(img, "SOBEL")


def filtroEmboss(R, G, B, offset):
    with open('emboss.txt', 'r') as f:
        filtro = np.array([[int(num) for num in line.split()]
                          for line in f])

    canal_r = convolve2d(R, filtro, mode='same', boundary='fill', fillvalue=0)
    canal_g = convolve2d(G, filtro, mode='same', boundary='fill', fillvalue=0)
    canal_b = convolve2d(B, filtro, mode='same', boundary='fill', fillvalue=0)

    canal_r += offset
    canal_g += offset
    canal_b += offset

    img = np.dstack((canal_r, canal_g, canal_b))
    img = np.clip(img, 0, 255)

    criarFigura(img, "EMBOSS")


def filtroMediana(image, tam_mascara):
    img = np.zeros_like(image)

    img_exp = np.pad(
        image, ((tam_mascara//2,), (tam_mascara//2,), (0,)), mode='constant')

    # Percorrer cada pixel da imagem
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            filtro = img_exp[i:i+tam_mascara, j:j+tam_mascara, :]
            mediana = np.median(filtro, axis=(0, 1))
            img[i, j, :] = mediana

    criarFigura(img, "MEDIANA")


def main():
    # Abre a imagem
    img = cv2.imread('imagem.tif')
    # img = cv2.imread('DancingInWater.jpg')

    # inverter a ordem dos canais bgr -> rgb
    img_rgb = img[:, :, ::-1]

    # Separa os canais RGB
    R = img_rgb[:, :, 0]
    G = img_rgb[:, :, 1]
    B = img_rgb[:, :, 2]

    criarFigura(img_rgb, 'RGB')
    imprimeCanaisRGB(R, G, B, img.shape[:2])
    converteRgbparaYiqParaRgb(R, G, B)
    negativoRgb(img_rgb)
    negativoEmY(R, G, B)
    filtroSoma(R, G, B)
    inicio11 = time.time()
    filtroMedia(R, G, B)
    fim11 = time.time()
    tempo_de_execucao11x11 = fim11 - inicio11
    inicio1 = time.time()
    filtroMediaDuplo(R, G, B)
    fim1 = time.time()
    tempo_de_execucao_duplo = fim1 - inicio1
    print("Tempo de execução 11x11: {:.2f} segundos".format(
        tempo_de_execucao11x11))
    print("Tempo de execução 1x11(11x1): {:.2f} segundos".format(
        tempo_de_execucao_duplo))
    filtroSobelH(R, G, B)
    filtroSobelV(R, G, B)
    filtroSobel(R, G, B)
    filtroEmboss(R, G, B, 128)
    filtroMediana(img_rgb, 9)

    plt.show()


main()
