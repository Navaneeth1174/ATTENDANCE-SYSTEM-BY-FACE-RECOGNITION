import cv2
import numpy as np
import os
import pickle

video = cv2.VideoCapture(0)  # 0 represents webcam
facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

face_data = []
i = 0
name = input("Enter your name: ")

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # CascadeClassifier uses grayscale images for detection
    faces = facedetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w, :]
        resized_img = cv2.resize(crop_img, dsize=(50, 50))
        if len(face_data) < 50 and i % 10 == 0:  # Collect 50 face samples
            cv2.putText(frame, str(len(face_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
            face_data.append(resized_img)

        cv2.imshow("Frame", frame)
        k = cv2.waitKey(1)
        if len(face_data) == 50:
            break
    i += 1
    if len(face_data) == 50:
        break

video.release()
cv2.destroyAllWindows()


# Saving dataset

face_data = np.array(face_data)
face_data = face_data.reshape(-1, 50 * 50 * 3)

if not os.path.exists('pickleDATA'):
    os.makedirs('pickleDATA')

if 'names.pkl' not in os.listdir('pickleDATA'):
    names = [name] * 50
    with open('pickleDATA/names.pkl', 'wb') as f:
        pickle.dump(names, f)
else:
    with open('pickleDATA/names.pkl', 'rb') as f:
        names = pickle.load(f)
        names = names + [name] * 50
    with open('pickleDATA/names.pkl', 'wb') as f:
        pickle.dump(names, f)

if 'faces_data.pkl' not in os.listdir('pickleDATA'):
    with open('pickleDATA/faces_data.pkl', 'wb') as f:
        pickle.dump(face_data, f)
else:
    with open('pickleDATA/faces_data.pkl', 'rb') as f:
        faces = pickle.load(f)
    faces = np.vstack((faces, face_data))
    with open('pickleDATA/faces_data.pkl', 'wb') as f:
        pickle.dump(faces, f)
