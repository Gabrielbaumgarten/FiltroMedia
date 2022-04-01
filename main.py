#===============================================================================
# Exemplo: segmentação de uma imagem em escala de cinza.
#-------------------------------------------------------------------------------
# Autor: Bogdan T. Nassu
# Universidade Tecnológica Federal do Paraná
#===============================================================================

from cProfile import label
from calendar import calendar
import sys
import timeit
import numpy as np
import cv2
import math

#===============================================================================

INPUT_IMAGE =  'img2.bmp'

# TODO: ajuste estes parâmetros!
NEGATIVO = False
THRESHOLD = 0.8
ALTURA_MIN = 10
LARGURA_MIN = 10
N_PIXELS_MIN = 400
# GRAY_SCALE = True
GRAY_SCALE = False

# números ímpares
WINDOW_SIZE = 9

#===============================================================================

def filtro_ingenuo(img):

    img_saida = np.zeros((img.shape[0], img.shape[1], 3), img.dtype)

    janela = math.floor(WINDOW_SIZE/2)

    if GRAY_SCALE:
        for linha in range(janela, img.shape[0] - janela ):
            for coluna in range(janela, img.shape[1] - janela ):
                soma = 0

                for window_row in range(-janela, janela + 1):
                    for window_column in range(-janela, janela + 1):
                        soma += img[linha+window_row][coluna+window_column][0]

                img_saida[linha][coluna] = soma/(WINDOW_SIZE*WINDOW_SIZE)
    else:
        for canal in range(3):
            for linha in range(janela, img.shape[0] - janela ):
                for coluna in range(janela, img.shape[1] - janela ):
                    soma = 0

                    for window_row in range(-janela, janela + 1):
                        for window_column in range(-janela, janela + 1):
                            soma += img[linha+window_row][coluna+window_column][canal]

                    img_saida[linha][coluna][canal] = soma/(WINDOW_SIZE*WINDOW_SIZE)

    return img_saida


#===============================================================================

def filtro_separavel(img):

    img_buffer = np.zeros((img.shape[0], img.shape[1], 3), np.float64)
    img_saida = np.zeros((img.shape[0], img.shape[1], 3), np.float64)

    janela = math.floor(WINDOW_SIZE/2)

    if GRAY_SCALE:
        for linha in range(janela, img.shape[0] - janela):
            for coluna in range(janela, img.shape[1] - janela):
                soma = 0

                for window_column in range(-janela, janela + 1):

                    soma += img[linha][coluna-window_column][0]

                img_buffer[linha][coluna] = soma

        for coluna in range(janela, img.shape[1] - janela):
            for linha in range(janela, img.shape[0] - janela):
                soma = 0

                for window_row in range(-janela, janela + 1):

                    soma += img_buffer[linha-window_row][coluna]

                img_saida[linha][coluna] = soma/(WINDOW_SIZE*WINDOW_SIZE)

    else:   
        for canal in range(3):     
            for linha in range(janela, img.shape[0] - janela):
                for coluna in range(janela, img.shape[1] - janela):
                    soma = 0

                    for window_column in range(-janela, janela + 1):

                        soma += img[linha][coluna-window_column][canal]

                    img_buffer[linha][coluna][canal] = soma

            for coluna in range(janela, img.shape[1] - janela):
                for linha in range(janela, img.shape[0] - janela):
                    soma = 0

                    for window_row in range(-janela, janela + 1):

                        soma += img_buffer[linha-window_row][coluna][canal]

                    img_saida[linha][coluna][canal] = soma/(WINDOW_SIZE*WINDOW_SIZE)



    return img_saida

#===============================================================================

def filtro_integral(img):

    img_buffer = np.zeros((img.shape[0], img.shape[1], 3), np.float64)
    img_saida = np.zeros((img.shape[0], img.shape[1], 3), np.float64)

    if GRAY_SCALE:
        for linha in range(img.shape[0]):
            for coluna in range(img.shape[1]):
                soma = img[linha][coluna][0]

                if coluna - 1 >= 0: 
                    soma += img_buffer[linha][coluna-1]

                img_buffer[linha][coluna] = soma

        for coluna in range(img.shape[1]):
            for linha in range(img.shape[0]):
                soma = img_buffer[linha][coluna]

                if linha - 1 >= 0: 
                    soma += img_buffer[linha-1][coluna]

                img_buffer[linha][coluna] = soma


        janela = math.floor(WINDOW_SIZE/2)

        for linha in range(img.shape[0] - janela ):
            for coluna in range(img.shape[1] - janela ):
                soma = 0

                if coluna-janela-1 >0:
                    soma -= img_buffer[linha+janela][coluna-janela-1]
                else:
                    soma -= img_buffer[linha+janela][coluna-janela]

                if linha-janela-1 >0:
                    soma -= img_buffer[linha-janela-1][coluna+janela]
                else:
                    soma -= img_buffer[linha-janela][coluna+janela]

                if coluna-janela-1 >0 and linha-janela-1 >0:
                    soma += img_buffer[linha-janela-1][coluna-janela-1]
                else:
                    soma += img_buffer[linha-janela][coluna-janela]

                soma += img_buffer[linha+janela][coluna+janela]

                img_saida[linha][coluna] = soma/(WINDOW_SIZE*WINDOW_SIZE)
    else:
        for canal in range(3):
            for linha in range(img.shape[0]):
                for coluna in range(img.shape[1]):
                    soma = img[linha][coluna][canal]

                    if coluna - 1 >= 0: 
                        soma += img_buffer[linha][coluna-1][canal]

                    img_buffer[linha][coluna][canal] = soma

            for coluna in range(img.shape[1]):
                for linha in range(img.shape[0]):
                    soma = img_buffer[linha][coluna][canal]

                    if linha - 1 >= 0: 
                        soma += img_buffer[linha-1][coluna][canal]

                    img_buffer[linha][coluna][canal] = soma


            janela = math.floor(WINDOW_SIZE/2)

            for linha in range(janela, img.shape[0] - janela ):
                for coluna in range(janela, img.shape[1] - janela ):
                    soma = 0

                    if coluna-janela-1 >0:
                        soma -= img_buffer[linha+janela][coluna-janela-1][canal]
                    else:
                        soma -= img_buffer[linha+janela][coluna-janela][canal]

                    if linha-janela-1 >0:
                        soma -= img_buffer[linha-janela-1][coluna+janela][canal]
                    else:
                        soma -= img_buffer[linha-janela][coluna+janela][canal]

                    if coluna-janela-1 >0 and linha-janela-1 >0:
                        soma += img_buffer[linha-janela-1][coluna-janela-1][canal]
                    else:
                        soma += img_buffer[linha-janela][coluna-janela][canal]

                    soma += img_buffer[linha+janela][coluna+janela][canal]

                    img_saida[linha][coluna][canal] = soma/(WINDOW_SIZE*WINDOW_SIZE)

    return img_saida

#===============================================================================

def main ():

    # Abre a imagem em escala de cinza.
    if GRAY_SCALE:
        img = cv2.imread (INPUT_IMAGE, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread (INPUT_IMAGE, cv2.IMREAD_COLOR)
    
    
    if img is None:
        print ('Erro abrindo a imagem.\n')
        sys.exit ()

    # É uma boa prática manter o shape com 3 valores, independente da imagem ser
    # colorida ou não. Também já convertemos para float32.
    if GRAY_SCALE:
        img = img.reshape ((img.shape [0], img.shape [1], 1))
    img = img.astype (np.float32) / 255


    img_ingenuo = filtro_ingenuo(img)
    cv2.imshow("imagemSaidaIngenuo", img_ingenuo)
    cv2.imwrite ('imagemSaidaIngenuo.png', img_ingenuo*255)

    img_separavel = filtro_separavel(img)
    cv2.imshow("imagemSaidaSeparavel", img_separavel)
    cv2.imwrite ('imagemSaidaSeparavel.png', img_separavel*255)
    
    img_integral = filtro_integral(img)
    cv2.imshow("imagemSaidaIntegral", img_integral)
    cv2.imwrite ('imagemSaidaIntegral.png', img_integral*255)

    img_out = cv2.blur(img, (WINDOW_SIZE, WINDOW_SIZE))
    cv2.imshow("imagemSaidaCV2", img_out)
    cv2.imwrite ('imagemSaidaCV2.png', img_out*255)

    
    cv2.waitKey ()
    cv2.destroyAllWindows ()


if __name__ == '__main__':
    main ()

#===============================================================================
