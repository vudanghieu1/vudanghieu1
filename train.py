import cv2
import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
import os


def main():
    # load all image to array training
    cells = []
    for i in range(10):
        cells.append([])
        arr = os.listdir("data/"+str(i))
        for j in arr:
            img = cv2.imread("data/"+str(i)+"/"+str(j), 0)
            cells[i].append(np.asarray(img))

    # list to aray
    x = np.array(cells)

    # convert to 1 row array
    train = x.reshape(-1, 400).astype(np.float32)

    # mark tag
    k = np.arange(10)
    train_labels = np.repeat(k, len(os.listdir("data//0")))[:, np.newaxis]

    # training data
    knn = cv2.ml.KNearest_create()
    knn.train(train, 0, train_labels)

    # load img to detect
    arr = []
    for i in range(12):
        arr.append(cv2.imread("img/"+str(i)+".jpg", 0))

    result = ''
    for j in arr:
        test1 = j.reshape(-1, 400).astype(np.float32)
        kq2 = knn.findNearest(test1, 5)[0]
        result += str(int(kq2))

    print(result)
