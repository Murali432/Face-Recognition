import cv2
import numpy as np
import os
import sqlite3
#Function to insert or update records in the database
def insertorupdate(Id, Name):
    conn = sqlite3.connect("sqlite.db")
    cursor = conn.execute("SELECT * FROM STUDENTS WHERE ID=?", (Id,))
    isRecordExist = cursor.fetchone() is not None
    if isRecordExist:
        conn.execute("UPDATE STUDENTS SET Name=? WHERE Id=?", (Name, Id))
    else:
        conn.execute("INSERT INTO STUDENTS (Id,Name) values(?,?)", (Id, Name))
    conn.commit()
    conn.close()

#Function to capture and save faces in the dataset
def capture_dataset(Id, face_cascade):
    cam = cv2.VideoCapture(0)
    sample_num = 0
    while(True):
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for(x, y, w, h) in faces:
            sample_num += 1
            cv2.imwrite(f"dataset/user.{Id}.{sample_num}.jpg",gray[y:y+h, x:x+w])
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.waitKey(100)
            cv2.imshow("Face", img)
            cv2.waitKey(1)
            if sample_num > 19:
                break
        if sample_num > 19:
            break
    cam.release()
    cv2.destroyAllWindows()

#Function to capture and save cropped faces
def capture_cropped_faces(face_cascade):
    cam = cv2.VideoCapture(0)
    img_id = 0
    while(True):
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for(x, y, w, h) in faces:
            img_id += 1
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))
            file_name_path = f"cropped_face/user.{Id}.{img_id}.jpg"
            cv2.imwrite(file_name_path, face)
            cv2.putText(frame, str(img_id), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imshow("cropped face", frame)
        if cv2.waitKey(1) == 13 or img_id == 20:
            break
    cam.release()
    cv2.destroyAllWindows()

#Function to detect and crop faces
def detect_and_crop_face(img, face_cascade):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    cropped_face = gray[y:y+h, x:x+w]
    return cropped_face

#Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#Get user ID and Name
num_users = int(input("Enter the number of users: "))
for i in range(num_users):
    Id = input(f"Enter User ID for user {i+1}: ")
    Name = input(f"Enter User Name for user {i+1}: ")
#Insert or update user record in the database
insertorupdate(Id, Name)
#Capture faces and save in the dataset
capture_dataset(Id, face_cascade)
#Capture cropped faces and save
capture_cropped_faces(face_cascade)