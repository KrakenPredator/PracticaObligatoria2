import cv2
import copy
import numpy as np

def ordenado(almacen):
    almCopy = copy.deepcopy(almacen)
    rtrn = []
    for ctr in almCopy:
        x, y, w, h = cv2.boundingRect(ctr)
        rtrn.append((x, y, w, h, (x, y), (x+w, y+h)))
    rtrn.sort()
    return rtrn

def filtradoRaw(imagen, almacen):
    almCopy = copy.deepcopy(almacen)
    rtrn = []
    for ctr in almCopy:
        x, y, w, h = cv2.boundingRect(ctr)
        area = cv2.contourArea(ctr)
        if w/h < .9:
            if area > 107:
                if area < 210:
                    print(area)
                    rtrn.append(((x, y), (x+w, y+h)))
    return rtrn


im = cv2.imread('testing_ocr/frontal_8.jpg', 0)
gauss = cv2.GaussianBlur(im, (3, 3), 0)
edgs = cv2.Canny(gauss, 100, 200)
_, thresh = cv2.threshold(im, 124, 255, 0)
_, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
imagen = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
#cajitas = ordenado(contours)
boxes = filtradoRaw(im, contours)
for bx in boxes:
    siz = bx[0]
    ider = bx[1]
    cv2.rectangle(imagen, siz, ider, (0, 255, 0), 2)

## Refinar el filtro para que no de tantos datos erróneos, buscar la forma de hacerlo en un solo if

cv2.imshow('boxes', imagen)
cv2.waitKey(0)

##<>
