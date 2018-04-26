import cv2
import glob
import numpy as np

ESCALA_x = 134
ESCALA_Y = 33
N_PX = 4422
ESCALA = 4
GREEN = (0, 255, 0)



def get_bounding(contornos):
    list = []
    for ctr in contornos:
        x, y, w, h = cv2.boundingRect(ctr)
        list.append((x, y, w, h, (x*ESCALA, y*ESCALA), ((x+w)*ESCALA, (y+h)*ESCALA), ctr))
    list.sort()
    return list


def dibujar_rectangulos(image):
    _, contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rectangulos = get_bounding(contours)
    inageR, boxes = filter(rectangulos, image)
    salidas = []
    for bx in boxes:
        x, y, w, h, _, _, _ = bx
        salidas.append(image[y:y+h, x:x+w])
    return inageR, salidas


def filter(almacen, image):
    outputHs = []
    boxes = []
    output = []
    imagec = cv2.resize(image, (0, 0), fx=ESCALA, fy=ESCALA)
    for bx in almacen:
        _, _, w, h, i, d, ctr = bx
        prop = w / h
        if prop < 1:
            area = cv2.contourArea(ctr)
            if 20 < area < 500:
                boxes.append(bx)
                outputHs.append(h)
    outputH = np.array(outputHs)
    if np.std(outputH) != 0:
        outputH = outputH[abs(outputH - np.mean(outputH)) < 1 * np.std(outputH)]
    else:
        outputH = outputHs
    for i in range(len(outputH)):
        altura = outputH[i]
        for j in range(len(boxes)):
            _, _, w, h, i, d, ctr = boxes[j]
            if altura == h:
                output.append(boxes[j])
                boxes.remove(boxes[j])
                break
    return imagec, output


def recortar_mat(image):
    image_out = 0
    mat_cascacade = cv2.CascadeClassifier("haar/matriculas.xml")
    matriculas = mat_cascacade.detectMultiScale(image)
    for (x, y, w, h) in matriculas:
        image_out = image[y:y+h, x:x+w]
    return image_out


def preprocess(image, umbral):
    _, thresh = cv2.threshold(image, umbral, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, thresh = cv2.threshold(thresh, umbral, 255, 0)
    return thresh

def main():
    for file in glob.glob('testing_ocr/*.jpg'):
        name = file.split('\\')
        archivo_name = name[len(name) - 1]
        file_image = cv2.imread(file, 0)
        rectangulos, boxes = matricula(file_image)
        cv2.imwrite('salida/' + archivo_name, rectangulos)


def matricula(file_image):
    matricula = recortar_mat(file_image)
    # matricula = cv2.resize(matricula, (ESCALA_x, ESCALA_Y))
    brillo_total = 0
    for fila in matricula:
        for pixel in fila:
            brillo_total += pixel
    brillo_medio = brillo_total // N_PX
    if brillo_medio < 150:
        if brillo_medio > 115:
            if brillo_medio > 127:
                if brillo_medio == 148:
                    matricula = preprocess(matricula, 13)
                else:
                    matricula = preprocess(matricula, 123)
            else:
                matricula = preprocess(matricula, 113)
        else:
            if brillo_medio > 90:
                if brillo_medio > 104:
                    matricula = preprocess(matricula, 103)
                else:
                    matricula = preprocess(matricula, 93)
            else:
                matricula = preprocess(matricula, 80)
    else:
        matricula = preprocess(matricula, 180)

    rectangulos, boxes = dibujar_rectangulos(matricula)
    return rectangulos, boxes

#main()