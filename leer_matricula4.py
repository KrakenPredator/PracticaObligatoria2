import cv2
import copy
import numpy as np
import glob

RED = (0, 0, 255)
GREEN = (0, 255, 0)
ESCALA = 4


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
    listC = set()
    maxh = 0
    for elem in lista:
        h = elem[3]
        if maxh < h:
            maxh = h

    for elem in lista:
        dif = np.abs(maxh-elem[3])
        if dif < 3:
            listC.add(elem)
    return listC

def ordenado(almacen):
    rtrn = {}
    for ctr in almacen:
        x, y, w, h = ctr
        if w/h < 1:
            factor = 3
            add = (x, y, w, h, (x*ESCALA, y*ESCALA), (x*ESCALA+w*ESCALA, y*ESCALA+h*ESCALA))
            if y//factor in rtrn:
                rtrn[y//factor].append(add)
            else:
                rtrn[y//factor] = [add]
    return rtrn


def ordenado_tamano(almacen):
    almc = []
    for ctr in almacen:
        almc.append(ctr)
    almc = son_igual_altura(almc)
    print(len(almc))
    rtrn = []
    for i in range(len(almc)):
        max = (0, 0, 0, 0)
        for ctr in almc:
            x, y, w, h = ctr
            print(w/h)
            if w/h < 1:
                multb = max[2]*max[3]
                multc = w*h
                if multb < multc:
                    max = ctr
        x, y, w, h = max
        print(max)
        add = (x, y, w, h, (x * ESCALA, y * ESCALA), (x * ESCALA + w * ESCALA, y * ESCALA + h * ESCALA))
        rtrn.append(add)
        if max != (0, 0, 0, 0):
            almc.remove(max)
    return rtrn


def filtradoMapa(almacen):
    rtrn = []
    max = 0
    for key in almacen.keys():
        lista = almacen[key]
        listC = son_igual_altura(lista)
        if len(listC) == 7:
            max = len(listC)
            rtrn = listC
            return rtrn
    return rtrn


def mostrarTodos(almacen):
    almCopy = copy.deepcopy(almacen)
    rtrn = []
    for ctr in almCopy:
        x, y, w, h = ctr
        rtrn.append((x, y, w, h, (x*ESCALA, y*ESCALA), (x*ESCALA+w*ESCALA, y*ESCALA+h*ESCALA)))
    return rtrn



for file in glob.glob('testing_ocr/*.jpg'):
    name = file.split('\\')
    fim = cv2.imread(file, 0)
    matricula = recortar_mat(fim)
    (x, y, w, h) = matricula
    #quitar los corchetes para ver la imagen entera
    im = fim[y: y+h, x:x+w]
    #im = cv2.GaussianBlur(im, (5, 5), 0)
    #cambiar nombres thresh2 por thresh para probar entre golbal y adaptive
    _, thresh = cv2.threshold(im, 125, 255, cv2.THRESH_BINARY)
    adap_thresh = cv2.adaptiveThreshold(im, 255.0, 1, 1, 11, 2)
    #_, thresh = cv2.threshold(im, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    #detected_edges = cv2.Canny(detected_edges, 100, 300, apertureSize=3)
    #dst = cv2.bitwise_and(im, im, mask=detected_edges)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    imagen = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    imagen = cv2.resize(imagen, (0, 0), fx=ESCALA, fy=ESCALA)
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
    cv2.imwrite('salida/' + name[len(name)-1], imagen)
    cv2.imwrite('salida/th' + name[len(name)-1], thresh)
    ## Refinar el filtro para que no de tantos datos erróneos, buscar la forma de hacerlo en un solo if


##<>
