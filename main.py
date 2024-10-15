import cv2
import pickle
import numpy as np
import os


video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
facedetect = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

faces_data = []
i = 0
name = input("Enter the name of the new person: ")


while True:
    ret, frame = video.read()
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        crop_image = frame[y:y+h, x:x+w, :]
        resized_image = cv2.resize(crop_image, (50, 50))

        
        if len(faces_data) < 50 and i % 10 == 0:
            faces_data.append(resized_image)
        i += 1

        
        cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 250), 1)

    cv2.imshow('frame', frame)
    k = cv2.waitKey(1)
    if k == ord('q') or len(faces_data) == 50:
        break

video.release()
cv2.destroyAllWindows()


faces_data = np.asarray(faces_data)
faces_data = faces_data.reshape(50, -1)


if 'data' not in os.listdir():
    os.mkdir('data')


if 'name.pkl' not in os.listdir('data/'):
    names = [name] * 50
    with open('data/name.pkl', 'wb') as f:
        pickle.dump(names, f)
else:
    with open('data/name.pkl', 'rb') as f:
        names = pickle.load(f)
    
    names.extend([name] * 50)
    with open('data/name.pkl', 'wb') as f:
        pickle.dump(names, f)


if 'faces_data.pkl' not in os.listdir('data/'):
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces_data, f)
else:
    with open('data/faces_data.pkl', 'rb') as f:
        faces = pickle.load(f)

    
    faces = np.vstack((faces, faces_data))

    
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces, f)

print(f"New data for {name} has been successfully added.")
