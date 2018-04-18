import cv2
import numpy as np
import glob
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

vector = np.zeros((9250, 100), dtype=np.float32)
E = []
for i in range(0,36):
    for j in range(0,250):
        E.append(i)


def leer_caracteres():
    tag = 1
    for file in glob.glob('training_ocr/*.jpg'):
        image_in = cv2.imread(file, 0)
        _, thresh = cv2.threshold(image_in, 150, 255, 0)
        _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        image_in = cv2.drawContours(image_in, contours, -1, color=(0, 255, 0), thickness=1)
        image_in = cv2.resize(image_in, (100, 100))
        matrix = np.reshape(image_in, (1, 100))
        vc = matrix.flatten()
        for x in range(len(vc)):
            vector[tag][x] = vc[x]
        cv2.imwrite('salida/' + str(tag) + '.jpg', image_in)
        tag += 1
    ldl = LDA()
    ldl.fit(vector, E)
    CR = ldl.transform(vector)


leer_caracteres()