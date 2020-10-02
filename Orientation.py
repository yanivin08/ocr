import cv2
import numpy as np
import pytesseract
import os

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\yaniv\AppData\Local\Tesseract-OCR\tesseract.exe'

per = 25
pixelThreshold=500
orb = cv2.ORB_create(100)

cImg = cv2.imread('IMG/Sample BOL.png')

#Get all the model list
path = 'Model'
myModelList = os.listdir(path)

for x,y in enumerate(myModelList):
    model = cv2.imread(path + "/" + y)

    if cImg.shape != model.shape:
        #resize current image to the size of model
        h,w,c = model.shape
        rImg = cv2.resize(cImg, (w,h))
        kp2, des2 = orb.detectAndCompute(rImg,None)
    else:
        kp2, des2 = orb.detectAndCompute(cImg,None)

    kp1, des1 = orb.detectAndCompute(model,None)
    
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    
    matches = bf.knnMatch(des1, des2, k=2)
    
    good = []

    for m , n in matches :
        if m.distance < 0.7 * n.distance:
            good.append([m])

    print(y + ": " + str(len(good)))

    if cImg.shape != model.shape:
        final_img = cv2.drawMatchesKnn(model,kp1,rImg,kp2,good,None)
    else:
        final_img = cv2.drawMatchesKnn(model,kp1,cImg,kp2,good,None)
        
    final_img = cv2.resize(final_img, (1500,1000))

    cv2.imshow(y,final_img)

