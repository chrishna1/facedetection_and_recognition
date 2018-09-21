import gspread
from oauth2client.service_account import ServiceAccountCredentials
import cv2
import numpy as np
import sqlite3
import os
from PIL import Image

scope= ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']
creds=ServiceAccountCredentials.from_json_keyfile_name('attendance-b4a9012c7c2d.json',scope)
client=gspread.authorize(creds)
sheet=client.open('attendance').sheet1


fontface = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 1
fontcolor = (255, 0, 0)

#recognizer = cv2.createLBPHFaceRecognizer()
#detector= cv2.CascadeClassifier("haarcascade_frontalface_default.xml");
recognizer = cv2.face.LBPHFaceRecognizer_create()
path = 'dataSet'

def getImagesWithId(path):
    #get the path of all the files in the folder
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
    #create empth face list
    faces=[]
    #create empty ID list
    Ids=[]
    #now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        #loading the image and converting it to gray scale
        faceImg = Image.open(imagePath).convert('L')
        #Now we are converting the PIL image into numpy array
        faceNp=np.array(faceImg,'uint8')
        #getting the Id from the image
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        #faces=detector.detectMultiScale(imageNp)
        #If a face is there then append that in the list as well as Id of it
        faces.append(faceNp)
        #print(Id)
        Ids.append(Id)
        cv2.imshow("training",faceNp)
        cv2.waitKey(10)
    return Ids, faces


Ids, faces = getImagesWithId(path)
recognizer.train(faces, np.array(Ids))

def getProfile(id):
    conn = sqlite3.connect("facebase.db")
    cmd = "SELECT * FROM People WHERE ID = " +str(id)
    cursor = conn.execute(cmd)
    profile = None
    for row in cursor:
        profile = row
    conn.close()
    return profile



faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)
#rec  = cv2.face.LBPHFaceRecognizer_create()

#rec.read("recognizer\\trainingData.yml")

id = 0

while(True):
    ret,img = cam.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h) ,(0,0,255),2)
        id, conf = recognizer.predict(gray[y:y+h,x:x+w])
        if(id==1):
            print('anupam')
        elif(id==2):
            print('sandeep')
        else:
            print('unknown')
##        if(id==1):
##            cv2.circle(img,(x,y),10,(0,0,255),-1)
##        if(id ==2):
##            cv2.circle(img,(x,y),10,(255,0,0),-1)

        profile = getProfile(id)
        if(profile!=None):
            cv2.putText(img,str(profile[1]),(x,y+h+30),fontface,fontscale,fontcolor)
            cv2.putText(img,str(profile[0]),(x,y+h+50),fontface,fontscale,fontcolor)
            #cv2.putText(img,str(profile[3]),(x,y+h+70),fontface,fontscale,fontcolor)
            #cv2.putText(img,str(profile[4]),(x,y+h+90),fontface,fontscale,fontcolor)
            while (profile[0]):
                row = [profile[0],profile[1]]
                index=2
                sheet.insert_row(row, index)
                break
    cv2.imshow("Face",img)
    if(cv2.waitKey(1) == ord('q')):
        break

cam.release()
cv2.destroyAllWindows()
