import numpy as np
import cv2
from PIL import Image
import sqlite3
import pickle
recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.read('recognizer/trainingData.yml')
cascadePath="Classifiers/face.xml"
faceCascade=cv2.CascadeClassifier(cascadePath);
path='dataSet'
def getProfile(id):
    conn=sqlite3.connect("facebase.db")
    cmd="SELECT ID from People"+str(id)
    cursor=conn.execute(cmd)
    profile=None
    for row in cursor:
        profile=row
    conn.close()
    return profile
cam= cv2.VideoCapture(0)
font=cv2.cv.InitFont(cv2.cv_Font_HERSHEY_SIMPLEX,1,1,0,1,1)


while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
####    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5,minSize(100,100),flags=cv2.C)
    for (x,y,w,h) in faces:
        id,conf=recognizer.predict(gray[y:y+h,x:x+w])
                                   
        #cv2.imwrite("dataSet/User."+str(id)+ "." +str(sampleNum)+".jpg", gray[y:y+h,x:x+w])

        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        profile=getProfile(id)
        if(profile!=None):
            cv2.putText(img, str(Id), (x,y+h), fontface, fontscale, fontcolor) 
            cv2.cv.putText(cv2.cv.fromarray(im),str(profile[1]),(x,y+h+30),font,225)
        
            cv2.cv.putText(cv2.cv.fromarray(im),str(profile[2]),(x,y+h+60),font,225)
            cv2.cv.putText(cv2.cv.fromarray(im),str(profile[3]),(x,y+h+90),font,225)
            cv2.cv.putText(cv2.cv.fromarray(im),str(profile(4)),(x,y+h+120),font,225)
            cv2.cv.putText(cv2.cv.fromarray(im),str(profile(5)),(x,y+h+150),font,225)
                    
    cv2.imshow('im',im)
    cv2.waitKey(1) 
        
    
cam.release()
cv2.destroyAllWindows()
