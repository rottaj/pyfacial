import cv2
import numpy as np

cap = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
eyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml')
while True:
    ret, frame = cap.read()
    print(frame, type(frame))
    print(type(cap), ret)
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    cv2.imshow('Input', frame)
    cv2.waitKey(1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30,30)
    )
    eyes = eyeCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(5,5)
    )
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        for(x,y,w,h) in eyes:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.imshow('Faces Found', frame)
            cv2.waitKey(3)


cap.release()
cv2.destroyAllWindows()
