import cv2
import numpy as np

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
img = cv2.imread('IMG_0256.jpeg', 0)
cv2.imshow('image', img)
cv2.waitKey(0)
print(type(img))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = faceCascade.detectMultiScale(
   gray,
   scaleFactor=1.1,
   minNeighbors=5,
   minSize=(30,30)
)
for(x,y,w,h) in faces:
    cv2.rectangle(img, (x,y), (x+w, y+h), (225,0,0), 2)
cv2.imshow('image', img)
cv2.waitKey(0)

