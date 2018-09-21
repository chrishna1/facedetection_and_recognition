import numpy as np
import cv2
import sqlite3
##
##
cam = cv2.VideoCapture(0)
detector= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
##def insertOrUpdate(Id,Name):
##    conn=sqlite3.connect("facebase.db")
##    cmd="SELECT * FROM People WHERE ID=?"+str(Id)
##    cursor=conn.execute(cmd)
##    isRecordExist=0
##    for row in cursor:
##        isRecordExist=1
##    if(isRecordExist==1):
##        cmd="UPDATE People SET Name" +str(Name) +"WHERE ID=" +str(Id)
##    else:
####        cmd="INSERT INTO People(ID,NAME) Values("+str(Id) +","+str(Name)+")"
##    conn.execute(cmd)
##    conn.commit()
##    conn.close()
##    
##import sqlite3
##import cv2
##import numpy as np
##import urllib.request as ur
##faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def insertOrUpdate(ID,Name):
    conn=sqlite3.connect("FaceBase.db")
    cursor=conn.execute('SELECT * FROM People WHERE id=?',[str(ID)])
    
    doesRecordExist=0
    for row in cursor:
        doesRecordExist=1
    if(doesRecordExist==1):
        cmd="UPDATE People SET Name="+str(Name)+" WHERE Id="+str(ID)
    else:
        conn.execute('INSERT INTO People(ID,Name) Values(?,?)',[str(ID),str(Name)])
        
    conn.commit()
    conn.close()




     
id= raw_input('enter user id')
Name= raw_input('enter user name')
insertOrUpdate(id,Name)


sampleNum = 0

while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        sampleNum = sampleNum + 1
        
        cv2.imwrite("dataSet/User."+str(id)+ "." +str(sampleNum)+".jpg", gray[y:y+h,x:x+w])

        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        
        cv2.waitKey(100)
    cv2.imshow('Face',img)
    cv2.waitKey(1) 
        
    if(sampleNum>20):
        break
cam.release()
cv2.destroyAllWindows()
