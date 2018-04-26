import cv2
import copy
import numpy as np
import glob

def redimensionar():
    for file in glob.glob('training_ocr/*.jpg'):

        name = file.split('\\')
        image = cv2.imread(file, 0)

        w, h = image.shape
        max_d = max(w, h)
        new_img = np.zeros((max_d, max_d), np.uint8)
        new_img.fill(255)

        for x in range(w):
            for y in range(h):
                k = image[x, y]
                new_img[x, y] = k

        new_img = cv2.resize(new_img,(10,10))
        cv2.imwrite('salida_ocr/'+name[len(name)-1], new_img)

        '''
        new_w = w//min
        new_h = h//min

        max = cv2.max(new_w, new_h)
        min = cv2.min(new_w, new_h)

        other = (min*10)//max

        if new_w == 1 :
            final_w = other
            final_h = 10
        else:
            if new_h == 1 :
                final_h == other
                final_w == 10


        imagen = cv2.resize(image, (0, 0), final_w, final_h)

        cv2.imshow('escalada',imagen)

        '''

redimensionar()