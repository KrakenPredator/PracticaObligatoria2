import cv2
import copy
import numpy as np
import external
import glob

RED = (0, 0, 255)
GREEN = (0, 255, 0)
ESCALA = 2


def recortar_mat(image):
    mat_cascacade = cv2.CascadeClassifier("haar/matriculas.xml")
    matriculas = mat_cascacade.detectMultiScale(image)
    for (x, y, w, h) in matriculas:
        recorte = (x, y, w, h)
    return recorte


def sacarBounding(contornos):
    list = []
    for ctr in contornos:
        x, y, w, h = cv2.boundingRect(ctr)
        list.append((x, y, w, h))
    list.sort()
    return list


def son_igual_altura(lista):
    listC = []
    actual = 0
    for elem in lista:
        if actual == 0:
            actual = elem[3]
            listC.append(elem)
        else:
            dif = np.abs(actual-elem[3])
            print(dif)
            actual = elem[3]
            if dif < 15:
                listC.append(elem)
    return listC

def ordenado(almacen):
    rtrn = {}
    for ctr in almacen:
        x, y, w, h = ctr
        if w/h < 1:
            factor = 3
            if y//factor in rtrn:
                rtrn[y//factor].append((x, y, w, h, (x*ESCALA, y*ESCALA), (x*ESCALA+w*ESCALA, y*ESCALA+h*ESCALA)))
            else:
                rtrn[y//factor] = [(x, y, w, h, (x*ESCALA, y*ESCALA), (x*ESCALA+w*ESCALA, y*ESCALA+h*ESCALA))]
    return rtrn


def filtradoMapa(almacen):
    rtrn = []
    max = 0
    for key in almacen.keys():
        lista = almacen[key]
        listC = son_igual_altura(lista)
        if len(listC) > max:
            max = len(listC)
            rtrn = listC
    return rtrn


def mostrarTodos(almacen):
    almCopy = copy.deepcopy(almacen)
    rtrn = []
    for ctr in almCopy:
        x, y, w, h = ctr
        rtrn.append((x, y, w, h, (x*ESCALA, y*ESCALA), (x*ESCALA+w*ESCALA, y*ESCALA+h*ESCALA)))
    return rtrn


joder = 1
for file in glob.glob('testing_ocr/*.jpg'):
    fim = cv2.imread(file, 0)
    matricula = recortar_mat(fim)
    (x, y, w, h) = matricula
    #quitar los corchetes para ver la imagen entera
    im = fim[y: y+h, x:x+w]
    gauss = cv2.GaussianBlur(im, (3, 3), 0)
    edgs = cv2.Canny(gauss, 100, 200)
    #cambiar nombres thresh2 por thresh para probar entre golbal y adaptive
    _, thresh2 = cv2.threshold(im, 150, 255, 0)
    _, thresh = cv2.threshold(im, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    imagen = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    tam = imagen.shape
    imagen = cv2.resize(imagen, (0, 0), fx=2, fy=2)
    listaCajas = sacarBounding(contours)
    cosas = ordenado(listaCajas)
    boxes = filtradoMapa(cosas)
    i = 0
    for bx in boxes:
        siz = bx[4]
        ider = bx[5]
        color = GREEN
        i+=1
        cv2.putText(imagen, str(i), bx[4], cv2.FONT_HERSHEY_PLAIN, 1, color, 1)
        cv2.rectangle(imagen, siz, ider, color, 2)
    cv2.imwrite('salida/' + str(joder) + 'thresh.jpg', thresh)
    cv2.imwrite('salida/'+str(joder)+'.jpg', imagen)
    joder += 1
    ## Refinar el filtro para que no de tantos datos erróneos, buscar la forma de hacerlo en un solo if


##<>
