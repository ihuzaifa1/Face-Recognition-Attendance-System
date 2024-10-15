from sklearn.neighbors import KNeighborsClassifier

import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime


video = cv2.VideoCapture(0 , cv2.CAP_DSHOW)
facedetect = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')



with open('data/name.pkl', 'rb') as f:
    Labels = pickle.load(f)
with open('data/faces_data.pkl', 'rb') as f:
    Faces = pickle.load(f)


cleaned_labels = []
for label in Labels:
    if isinstance(label, list):
        
        for item in label:
            if isinstance(item, str):
                cleaned_labels.append(item)  
    elif isinstance(label, str):
        
        cleaned_labels.append(label)


for index, label in enumerate(cleaned_labels):
    if not isinstance(label, str):
        print(f"Still a problem at index {index}: {label} (type: {type(label)})")


Labels = np.array(cleaned_labels)


print(f"Cleaned Labels shape: {Labels.shape}")


if Faces.shape[0] != len(cleaned_labels):
    print(f"Mismatch: Faces has {Faces.shape[0]} samples, but Labels has {len(cleaned_labels)} labels.")
    
    
    cleaned_labels = cleaned_labels[:Faces.shape[0]]


Labels = np.array(cleaned_labels)


print(f"Cleaned Faces shape: {Faces.shape}")
print(f"Cleaned Labels shape: {Labels.shape}")


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(Faces,Labels)

ts = time.time()
date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")


column_names = ['Name',str(date)]

while True:
    ret, frame = video.read()
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))

    for(x,y,w,h) in faces:
        crop_image = frame[y: y+h, x:x + w, :]
        resized_image = cv2.resize(crop_image, (50,50)).flatten().reshape(1,-1)

        output = knn.predict(resized_image)
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
        
        exist = os.path.isfile('attendence/attendence_' +date + '.csv')


        cv2.putText(frame, str(output[0]),(x,y-15),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),1)
        cv2.rectangle(frame, (x,y), (x+w, y+h),(50,50,250),1)
        attendence = [str(output[0]),str('P')]

    cv2.imshow('frame',frame)
    k = cv2.waitKey(1)
    if k == ord('n'):
        if exist:
                with open('attendence/attendence_' +date + '.csv','+a') as csvfile:
                      writer = csv.writer(csvfile)
                      writer.writerow(attendence)
                csvfile.close()
        else:
                with open('attendence/attendence_' +date + '.csv','+a') as csvfile:
                      writer = csv.writer(csvfile)
                      writer.writerow(column_names)
                      writer.writerow(attendence)
                csvfile.close()
              
    if k == ord('q'):
        break
video.release()
cv2.destroyAllWindows()



