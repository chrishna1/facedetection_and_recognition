
import cv2,os
import numpy as np
from PIL import Image

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
recognizer.write('recognizer/trainingData.yml')
cv2.destroyAllWindows()
