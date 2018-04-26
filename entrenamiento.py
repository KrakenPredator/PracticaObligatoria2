import cv2
import numpy as np
import glob
import new_matriculas
import os
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

C = np.zeros((9251, 100), dtype=np.float32)
E = []
for i in range(0,37):
    for j in range(0,250):
        E.append(i)
E.append(i)
E = np.array(E, np.float32)
E = E.reshape((E.size, 1))


def filter_training(almacen, image):
    all = []
    for box in almacen:
        (x, y, w, h)= cv2.boundingRect(box)
        all.append((w * h, x, y, w, h))
    all.sort()
    if len(all) == 3:
        boxi = all[1]
        image_out = image[boxi[2]:boxi[2]+boxi[4], boxi[1]:boxi[1]+boxi[3]]

    else:
        if (len(all) == 2):
            boxi = all[1]
            image_out = image[boxi[2]:boxi[2] + boxi[4], boxi[1]:boxi[1] + boxi[3]]
        else:
            print(len(all)/2)
            boxi = all[len(all) // 2]
            image_out = image[boxi[2]:boxi[2] + boxi[4], boxi[1]:boxi[1] + boxi[3]]
    return image_out


def sacar_vc(image_in, name):
    _, thresh = cv2.threshold(image_in, 15, 255, 0)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    print(name)
    image_out = filter_training(contours, image_in)
    cv2.imwrite('salida_ocr/' + name, image_out)
    image_out = redimensionar(image_out)
    matrix = np.reshape(image_out, (1, 100))
    vc = matrix.flatten()
    return vc, image_out


def leer_caracteres():
    tag = 0
    for file in glob.glob('training_ocr/*.jpg'):
        image_in = cv2.imread(file, 0)
        name = file.split('\\')
        name = name[len(name) - 1]
        vc, im_out = sacar_vc(image_in, name)
        C[tag] = vc
        tag += 1
    np.savetxt('c_entrenado.data', C)
    np.savetxt('e_entrenado.data', E)


def redimensionar(image):
    w, h = image.shape
    max_d = max(w, h)
    new_img = np.zeros((max_d, max_d), np.uint8)
    new_img.fill(255)
    image = cv2.threshold(image, 126, 255, 0)[1]
    for x in range(w):
        for y in range(h):
            k = image[x, y]
            new_img[x, y] = k

    new_img = cv2.resize(new_img,(10,10))
    return new_img


samples = np.loadtxt('c_entrenado.data', np.float32)
responses = np.loadtxt('e_entrenado.data', np.float32)

responses = responses.reshape((responses.size, 1))
path = os.getcwd()
modelo = cv2.ml.KNearest_create()
modelo.train(samples, cv2.ml.ROW_SAMPLE, responses)
for file in glob.glob('testing_ocr/*.jpg'):
    name = file.split('\\')
    name = name[len(name)-1]
    name_dir = name.split('.')
    name_dir = name_dir[0]
    new_path = path+"\\salidas_entrenamiento\\"+name_dir
    print(new_path)
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    imgTest = cv2.imread(file, 0)
    matricula, recortes = new_matriculas.matricula(imgTest)
    t = 0
    os.chdir(new_path)
    for m in recortes:
        t+=1

        img = redimensionar(m)

        mat = np.reshape(img, (1, 100))
        mat = np.asarray(mat, dtype=np.float32)
        val, res = modelo.predict(mat)
        ret, resultado, vecinos, dist = modelo.findNearest(mat, k=3)  # Obtenemos los resultados para los k vecinos mas cercanos
        correctosK = np.count_nonzero(resultado == E)
        print(str(resultado), str(vecinos), str(dist), str(t))
        img = cv2.resize(m, (300, 300))
        cv2.imwrite(str(t)+'-'+str(val)+".jpg", img)
    os.chdir(path)
