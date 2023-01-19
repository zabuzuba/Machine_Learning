import cv2
import glob
import os
import numpy as np
minValue = 70

os.mkdir('Y:/silars/archive/data/K/grey')
images_path = glob.glob('Y:/silars/archive/data/K/*.jpg')
i=0
for image in images_path:
    img = cv2.imread(image)
    gray_images = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray_images,(5,5),2)
   # cv2.imshow('Gray Images', gray_images)
    th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    cv2.imwrite('Y:/silars/archive/data/K/grey/image%02i.jpg' %i, res)
    i += 1
    cv2.waitKey(600)
    cv2.destroyAllWindows()