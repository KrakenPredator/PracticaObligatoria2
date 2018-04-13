import cv2
import numpy as np

im = cv2.imread('testing_ocr/frontal_31.jpg', 0)
rev, thresh = cv2.threshold(im, 80, 255, cv2.THRESH_BINARY)
zzzz, contours, hhhh= cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
imagen = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
almacenados = []
for idx in range(len(contours)):
    x, y, w, h = cv2.boundingRect(contours[idx])
    if (h in range(15, 28)) & (w in range(3, 18)):
        almacenados.append((x, y, w, h, x+w, y+h))
almacenados.sort()
contador = 7
matriculay = 0
for i in range(len(almacenados)-1):
    act = almacenados[i]
    sig = almacenados[i+1]
    dist = sig[1] - act[1]
    if contador != 0:
        if np.abs(dist) < 11:
            contador -= 1
            matriculay = act[1]
            print("matricula encontrada en ", matriculay)
            sizq = (act[0], act[1])
            ider = (act[4], act[5])
            cv2.rectangle(imagen, sizq, ider, (0, 255, 0), 2)
            cv2.putText(imagen, str(i), (act[0], act[1]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
        else:
            diferencia_mat = np.abs(matriculay - act[1])
            if diferencia_mat < 11:
                contador -= 1
                sizq = (act[0], act[1])
                ider = (act[4], act[5])
                cv2.rectangle(imagen, sizq, ider, (0, 255, 0), 2)
                cv2.putText(imagen, str(i), (act[0], act[1]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
            else:
                print(i, diferencia_mat)
                sizq = (act[0], act[1])
                ider = (act[4], act[5])
                cv2.rectangle(imagen, sizq, ider, (0, 0, 255), 2)
                cv2.putText(imagen, str(i), (act[0], act[1]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

#Refinar el filtro para que no de tantos datos erróneos, buscar la forma de hacerlo en un solo if
cv2.imshow('boxes', imagen)
cv2.waitKey(0)

##<>