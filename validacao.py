import numpy as np
import pandas as pd
import cv2
import os
import sys
import glob



from matplotlib import pyplot as plt
from skimage.morphology import skeletonize
from scipy.signal import find_peaks
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
#################################################
def colorDec(cor,canBlack = True):


    centroCores = {
        'Vermelho':[0,180],
        'Laranja': 10,
        'Amarelo':25,
        'Verde':50,
        'Azul':110,
        'Violeta':130
    }

    melhor_cor = None
    menor_dist = float('inf')
    hcor = cor[0]
    for nome, centro in centroCores.items():

        if nome == 'Vermelho':
          dist1 = np.linalg.norm(np.array(hcor) - np.array(centro[0]))
          dist2 = np.linalg.norm(np.array(hcor) - np.array(centro[1]))
          dist = min(dist1,dist2)

        else:
          dist = np.linalg.norm(
              np.array(hcor) - np.array(centro)
          )

        if dist < menor_dist:
            menor_dist = dist
            melhor_cor = nome

    if melhor_cor == 'Vermelho' or melhor_cor == 'Laranja':
            if (cor[1] < 70 and (30 < cor[2] < 60)) or (cor[1] < 20 and (40 < cor[2] < 100)):
                if cor[2] < 30 and canBlack:
                    melhor_cor = 'Preto'

                else:
                    melhor_cor = 'Cinza'

            elif cor[2] < 70:
                if cor[2] < 15 and canBlack:
                    melhor_cor = 'Preto'

                else:
                    melhor_cor = 'Marrom'
            
            elif cor[2] > 100:
                melhor_cor = 'Laranja'

    else:
            if (cor[1] < 70 and cor[2] < 60) or (cor[1] < 20 and cor[2] < 100):
                if cor[2] < 30 and canBlack:
                    melhor_cor = 'Preto'

                else:
                    melhor_cor = 'Cinza'

            elif cor[2] < 20 and canBlack:
                melhor_cor = 'Preto'

    # print(melhor_cor, menor_dist)
    return melhor_cor



def resistencia(cores):

  colorcode = {'Preto':0,
               'Marrom':1,
               'Vermelho':2,
               'Laranja':3,
               'Amarelo':4,
               'Verde':5,
               'Azul':6,
               'Violeta':7,
               'Cinza':8
               }
  holder = []

  for cor in cores:
    holder.append(colorcode[cor])


  resist1 = (holder[0]*10 + holder[1])*10**(holder[2])
  resist = formatRes(resist1)
  resistor = [resist,resist1]
  return resistor
  print(f'Seu resistor é de {resist} omhs! ')


def formatRes(r):
    if r < 1e3:
        return f'{r:.0f} Ω'
    elif r < 1e6:
        return f'{r/1e3:.1f} kΩ'
    elif r < 1e9:
        return f'{r/1e6:.1f} MΩ'
    else:
        return f'{r/1e9:.1f} GΩ'

#################################################
def main(img_path):

    ########################### ELIMINAÇÃO DE SOMBRA #################

    # res_num = input('Qual número do res: ')
    img = cv2.imread(img_path)
    # imgh = cv2.imread(f'/content/drive/MyDrive/Projeto - Resistores/Resistores/Resistor_{res_num}.png')
    # img = cv2.imread(f'/content/drive/MyDrive/Projeto - Resistores/ResHist/ResHist{res_num}.png')

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)

    v = cv2.GaussianBlur(v,(5,5),0)
    bg = cv2.GaussianBlur(v,(101,101),0)
    v2 = cv2.divide(v,bg,scale=255)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    v2 = clahe.apply(v2)
    final = cv2.merge((h,s,v2)) #Imagem com fundo muito claro para limiarização.

    ##################### LIMIARIZAÇÃO E MORFOLOGIA ##################

    lower = np.array([0, 0, 170])
    upper = np.array([180, 45, 255])
    background = cv2.inRange(final, lower, upper)
    mask = cv2.bitwise_not(background)

    # Fechamento e remoção de ruídos
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5,5)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        np.ones((3,3), np.uint8)
    )
    ########################### ENCONTRA CONTORNOS #########################

    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contorno_escolhido = max(contornos, key=cv2.contourArea) #Pega o contorno de maior área: Se a imagem respeitar as condições impostas, esse contorno sempre representará pelo menos o corpo do resistor
    mask = np.zeros(final.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contorno_escolhido], -1, 255, -1)

    ########################################################################
    radius = cv2.distanceTransform(mask, cv2.DIST_L2, 5).max()

    L =int( radius if radius%2 > 0 else radius+ 1 )
    print(L)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (L, L)
    )


    # Opening para eliminar as pernas
    dilate = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel)

    resistor = dilate

    #############################################################################

    resultado = cv2.bitwise_and(img, img, mask=resistor)

    ContRes, _ = cv2.findContours(resistor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rect = cv2.minAreaRect(max(ContRes, key=cv2.contourArea))
    (cx_rect, cy_rect), (w, h), angle = rect


    if w < h:
        angle = angle + 90

    M = cv2.getRotationMatrix2D((cx_rect, cy_rect), angle, 1.0)

    imgRotac = cv2.warpAffine(
        resultado,
        M,
        (resultado.shape[1], resultado.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )

    imgRotac2 = cv2.warpAffine(
        img,
        M,
        (resultado.shape[1], resultado.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )

    resistor = cv2.warpAffine(
        resistor,
        M,
        (resistor.shape[1], resistor.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    blur = cv2.GaussianBlur(imgRotac2, (11,11), 0)

    corpohsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    hc,sc,vc = cv2.split(corpohsv)

    LFaixas =int( L*0.2 if radius%2 > 0 else L*0.2+ 1 )

    kernelFaixas = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (LFaixas, LFaixas)
    )

    hcc = hc[resistor>0]
    histh = cv2.calcHist([hcc],[0],None, [179],[0,179])
    maxh = np.argmax(histh)


    maskhInv = cv2.inRange(hc, int(maxh - 12), int(maxh + 12))

    maskh = cv2.bitwise_and(
        cv2.bitwise_not(maskhInv),
        resistor
    )

    maskh = cv2.morphologyEx(
        maskh,
        cv2.MORPH_OPEN,
        kernelFaixas)



    corpo = cv2.bitwise_and(blur,blur,mask = maskhInv)
    corpohsv2 = cv2.cvtColor(corpo,cv2.COLOR_BGR2HSV)
    hb,sb,vb = cv2.split(corpohsv2)

    #Extração de cores claras que foram filtrada como corpo

    _,masks = cv2.threshold(sb,150,255,cv2.THRESH_BINARY)

    masks = cv2.morphologyEx(
        masks,
        cv2.MORPH_OPEN,
        kernelFaixas)

    maskf = cv2.bitwise_or(maskh,masks)

    gray = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_and(gray,gray,mask = maskhInv)

    _,maskgInv = cv2.threshold(gray,50,255,cv2.THRESH_BINARY)

    maskg = cv2.bitwise_and(
        cv2.bitwise_not(maskgInv),
        resistor
    )

    maskg = cv2.morphologyEx(
        maskg,
        cv2.MORPH_OPEN,
        kernelFaixas)

    maskff = cv2.bitwise_or(maskf,maskg)

    maskff = cv2.morphologyEx(
        maskff
        ,
        cv2.MORPH_OPEN,
        kernelFaixas)

    FaixasCont = imgRotac.copy()
    contornosF, _ = cv2.findContours(maskff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(FaixasCont, contornosF, -1, (0, 255, 0), 3)
    dados = []

    for contorno in contornosF:
        M = cv2.moments(contorno)

        if M["m00"] == 0:
            continue

        cx = M["m10"] / M["m00"]
        area = cv2.contourArea(contorno)

        dados.append({
            "contorno": contorno,
            "cx": cx,
            "area": area
        })

    # Marca contornos para remoção
    remover = set()

    for i in range(len(dados)):
        for j in range(i + 1, len(dados)):

            if abs(dados[i]["cx"] - dados[j]["cx"]) <= 5:

                # Remove o de menor área
                if dados[i]["area"] < dados[j]["area"]:
                    remover.add(i)
                else:
                    remover.add(j)

    # Reconstrói faixas sem os contornos removidos
    faixas = [
        dados[i]["contorno"]
        for i in range(len(dados))
        if i not in remover
    ]

    faixas = sorted(
        faixas,
        key=cv2.contourArea,
        reverse=True
    )[:3]



    # ORdena as Faixas para ler o resistor na ordem orreta
    ContRes, _ = cv2.findContours(resistor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    x_res, y_res, w_res, h_res = cv2.boundingRect(
        max(ContRes, key=cv2.contourArea)
    )

    xs = []

    # Centroide de cada faixa
    for faixa in faixas:
        M = cv2.moments(faixa)

        px = int(M["m10"] / M["m00"])
        xs.append(px)

    # Distância até a borda esquerda e direita do resistor
    d_esq = [x - x_res for x in xs]
    d_dir = [x_res + w_res - x for x in xs]

    # Faixas mais à esquerda e mais à direita
    argmin = np.argmin(xs)
    argmax = np.argmax(xs)

    # Faixa do meio
    meio = ({0, 1, 2} - {argmin, argmax}).pop()

    FdArg1 = [d_esq[argmin], d_dir[argmin]]
    FdArg2 = [d_esq[argmax], d_dir[argmax]]

    # Quem estiver mais próximo de uma extremidade é a primeira faixa
    if min(FdArg1) < min(FdArg2):
        ordem = [argmin, meio, argmax]
    else:
        ordem = [argmax, meio, argmin]

    FaixasOrdenadas = [faixas[i] for i in ordem]

    colors = []
    colors2 = []



    for Faixa in FaixasOrdenadas:

        M = cv2.moments(Faixa)
        px = int(M["m10"] / M["m00"])
        py = int(M["m01"] / M["m00"])

        janela = blur[py-2:py+3, px-2:px+3]

        # Converte toda a janela para HSV
        janela_hsv = cv2.cvtColor(janela, cv2.COLOR_BGR2HSV)

        # Média dos pixels HSV da janela
        media_hsv = np.mean(janela_hsv, axis=(0, 1))

        h, s, v = media_hsv

        colors.append([h, s, v])

        colors2.append(np.array([
            h,
            s,
            v
        ], dtype=np.float32))

    corFaixas = []
    print(colors2)
    for cor in colors2:

        corFaixas.append(colorDec(cor))


    if corFaixas.count('Preto') > 1:
        minBlack = min(
            cor[2]
            for nome, cor in zip(corFaixas, colors2)
            if nome == 'Preto'
        )

        colorHolder = colors2.copy()
        # colorHolder.pop(idx)
        # print(colorHolder)

        for idxCor, cor in enumerate(colorHolder):
            if corFaixas[idxCor] == 'Preto' and np.abs(cor[2] - minBlack) >= 5:
                corFaixas[idxCor] = colorDec(cor, canBlack=False)



    _, resNum = resistencia(corFaixas)
    return resNum


arquivo = './Resistores/ResValues.csv'

df = pd.read_csv(arquivo)

mapa_resistencias = dict(
    zip(df["Imagem"], df["Resistencia"])
)

imagens = glob.glob('./Resistores/*.png')

acertos = 0
erros = 0

for img_path in imagens:

    nome = os.path.basename(img_path)

    try:

        valor_detectado = main(img_path)

        if nome not in mapa_resistencias:
            print(f"{nome} não encontrado no CSV")
            continue

        valor_real = int(mapa_resistencias[nome])

        if valor_detectado == valor_real:
            acertos += 1
            print(f"✓ {nome}: {valor_detectado}")

        else:
            erros += 1
            print(
                f"✗ {nome}: "
                f"detectado={valor_detectado}, "
                f"esperado={valor_real}"
            )

    except Exception as e:
        erros += 1
        print(f"ERRO em {nome}: {e}")

total = acertos + erros

print("\n==========================")
print(f"Total    : {total}")
print(f"Acertos  : {acertos}")
print(f"Erros    : {erros}")

if total > 0:
    print(f"Acurácia : {100 * acertos / total:.2f}%")