import cv2
import numpy as np
import glob
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

C = np.zeros((9250, 100), dtype=np.float32)
E = []
for i in range(0,36):
    for j in range(0,250):
        E.append(i)


def leer_caracteres():
    tag = 1
    for file in glob.glob('training_ocr/*.jpg'):
        name = file.split('\\')
        image_in = cv2.imread(file, 0)
        _, thresh = cv2.threshold(image_in, 150, 255, 0)
        _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        image_in = cv2.drawContours(image_in, contours, -1, color=(0, 255, 0), thickness=1)
        image_in = cv2.resize(image_in, (10, 10))
        matrix = np.reshape(image_in, (1, 100))
        vc = matrix.flatten()
        for x in range(len(vc)):
            C[tag][x] = vc[x]
        cv2.imwrite('salida/out_' + name[len(name)-1] , image_in)
        tag += 1
    lda = LDA()
    lda.fit(C, E)
    CR = lda.transform(C)


leer_caracteres()