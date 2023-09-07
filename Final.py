import os
import cv2
import csv
import numpy as np
from datetime import datetime, timedelta
from keras.models import load_model
from PIL import Image as Img
from numpy import asarray, expand_dims
from keras_facenet import FaceNet
import gspread
import pickle
from google.oauth2.service_account import Credentials
import pandas as pd
import itertools
import firebase_admin
from firebase_admin import credentials, initialize_app, db
import shutil
import json
import mediapipe as mp
import tensorflow as tf


# Function to detect and crop faces in an image
def detect_and_crop_faces(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    cropped_faces = []
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cropped_face = image[y:y + h, x:x + w]
        cropped_faces.append(cropped_face)

    return cropped_faces

root_directory = 'images'

subdirectories = [name for name in os.listdir(root_directory) if os.path.isdir(os.path.join(root_directory, name)) and name.startswith('cut_')]
counter = 1
for subdirectory in subdirectories:
    counter=1
    subdirectory_path = os.path.join(root_directory, subdirectory)
    image_files = [name for name in os.listdir(subdirectory_path) if os.path.isfile(os.path.join(subdirectory_path, name)) and name.lower().endswith(('.jpg', '.jpeg', '.png'))]
    os.makedirs(os.path.join(root_directory,subdirectory[4:]))
    for image_file in image_files:
        image_path = os.path.join(subdirectory_path, image_file)

        image = cv2.imread(image_path)
        cropped_faces = detect_and_crop_faces(image)
        for cropped_face in cropped_faces:
            save_path = os.path.join(root_directory, subdirectory[len("cut_"):],f'{counter}.jpg')
            cv2.imwrite(save_path, cropped_face)
            counter += 1


for folder_name in os.listdir(root_directory):
    folder_path = os.path.join(root_directory, folder_name)
    
    if folder_name.startswith('cut_') and os.path.isdir(folder_path):
        shutil.rmtree(folder_path)


HaarCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
MyFaceNet = FaceNet()
parent_folder = 'images'
database = {}

if os.path.isfile("data.pkl"):
    with open("data.pkl", "rb") as myfile:
        database = pickle.load(myfile)

existing_folders = list(database.keys())

for folder_name in os.listdir(parent_folder):
    folder_path = os.path.join(parent_folder, folder_name)

    if not os.path.isdir(folder_path):
        continue

    if folder_name not in existing_folders:
        database[folder_name] = []

        for filename in os.listdir(folder_path):
            image_path = os.path.join(folder_path, filename)
            gbr1 = cv2.imread(image_path)

            if gbr1 is None:
                print(f"Failed to load image: {image_path}")
                continue

            wajah = HaarCascade.detectMultiScale(gbr1, 1.1, 4)

            if len(wajah) > 0:
                x1, y1, width, height = wajah[0]
            else:
                x1, y1, width, height = 1, 1, 10, 10

            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height

            gbr = cv2.cvtColor(gbr1, cv2.COLOR_BGR2RGB)
            gbr = Img.fromarray(gbr)
            gbr_array = asarray(gbr)

            face = gbr_array[y1:y2, x1:x2]

            face = Img.fromarray(face)
            face = face.resize((160, 160))
            face = asarray(face)

            face = expand_dims(face, axis=0)
            signature = MyFaceNet.embeddings(face)

            database[folder_name].append(signature)
            
deleted_folders = [folder for folder in existing_folders if folder not in os.listdir(parent_folder)]
for folder in deleted_folders:
    database.pop(folder, None)

if len(deleted_folders) > 0 or len(existing_folders) != len(os.listdir(parent_folder)):
    with open("data.pkl", "wb") as myfile:
        pickle.dump(database,myfile)
cred = Credentials.from_service_account_file('sheets.json')
scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
cred = cred.with_scopes(scope)
client = gspread.authorize(cred)
spreadsheet = client.open('Data_Log')
worksheet = spreadsheet.sheet1
fire = credentials.Certificate('firebase.json')
firebase_admin.initialize_app(fire, {'databaseURL': 'https://test--esp-default-rtdb.firebaseio.com/'})

logged_data = {}
threshold = 0.6
last_denied_timestamp = {}

csv_file = 'log.csv'
csv_header = ['Date', 'Time', 'Name', 'Access Status']

db.reference('test').set({})

access_names = set()
with open('peoples.json') as file:
    data = json.load(file)

for person in data["peoples"]:
    if person["access"] == "true" or person["access"] is True :
        access_names.add(person["id"])
print(access_names)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)
num_faces = 0

while num_faces < 1:
    _, gbr1 = cap.read()

    wajah = HaarCascade.detectMultiScale(gbr1, 1.1, 4)

    num_faces = len(wajah)

    cv2.putText(gbr1, "Waiting for faces...", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('res', gbr1)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

for x in itertools.count():
    _, gbr1 = cap.read()
    current_time = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")

    wajah = HaarCascade.detectMultiScale(gbr1, 1.1, 4)

    while cap.isOpened():        
        success, image = cap.read()
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)        
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                pass
            xco=[]
            yco=[]
            for id,lms in enumerate(hand_landmarks.landmark):
                ih,iw,ic = image.shape
                x,y = int(lms.x*iw),int(lms.y*ih)
                xco.append(x)
                yco.append(y)

            if yco[17]>yco[18]>yco[19]>yco[20]:
                p = 'open'
            else:
                p = 'close'
            if yco[5]>yco[6]>yco[7]>yco[8]:
                i = 'open'
            else:
                i = 'close'            
            if yco[13]>yco[14]>yco[15]>yco[16]:
                r = 'open'
            else:
                r = 'close'            
            if (p and i) and (p and r) and (i and r) == 'open':
                break
            else:
                pass

    if len(wajah) > 0:
        x1, y1, width, height = wajah[0]
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height

        gbr = cv2.cvtColor(gbr1, cv2.COLOR_BGR2RGB)
        gbr = Img.fromarray(gbr)
        gbr_array = asarray(gbr)

        face = gbr_array[y1:y2, x1:x2]

        face = Img.fromarray(face)
        face = face.resize((160, 160))
        face = asarray(face)

        face = expand_dims(face, axis=0)
        signature = MyFaceNet.embeddings(face)

        min_dist = 100
        identity = 'Unknown'
        access_status = 'Denied'

        for key, values in database.items():
            for value in values:
                dist = np.linalg.norm(value - signature)
                if dist < min_dist:
                    min_dist = dist
                    identity = key

        if min_dist <= threshold:
            if identity in access_names:
                cv2.putText(gbr1, f"Name: {identity}", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(gbr1, "Access Granted", (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                if (identity, current_time) not in logged_data:
                    if not any(
                            entry[0] == identity and datetime.strptime(current_time, "%d-%m-%Y_%H:%M:%S") - datetime.strptime(
                                    entry[1], "%d-%m-%Y_%H:%M:%S") < timedelta(seconds=30) for entry in logged_data.get(identity, [])):
                        access_status = 'Granted'
                        with open(csv_file, 'a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow([current_time.split('_')[0], current_time.split('_')[1], identity, access_status])
                        db.reference('test').update({'int': 1, 'string': identity})
                        worksheet.append_row([current_time.split('_')[0], current_time.split('_')[1], identity, access_status])
                        if identity not in logged_data:
                            logged_data[identity] = []
                        logged_data[identity].append((identity, current_time))
                        print(f"Access granted: {current_time.split('_')[0]}, {current_time.split('_')[1]}, {identity}")
                    else:
                        print(
                            f"Access already granted within the last minute: {current_time.split('_')[0]}, {current_time.split('_')[1]}, {identity}")
                else:
                    print(f"Access already logged: {current_time.split('_')[0]}, {current_time.split('_')[1]}, {identity}")
            else:
                cv2.putText(gbr1, f"Name: {identity}", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(gbr1, "Access Denied", (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                print(f"Access denied: {current_time.split('_')[0]}, {current_time.split('_')[1]}, {identity}")
                last_denied_time = last_denied_timestamp.get(identity)
                if last_denied_time is None or (datetime.strptime(current_time, "%d-%m-%Y_%H:%M:%S") - last_denied_time) >= timedelta(seconds=30):
                    with open(csv_file, 'a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([current_time.split('_')[0], current_time.split('_')[1], identity, access_status])
                    db.reference('test').update({'int': 0, 'string': identity})
                    worksheet.append_row([current_time.split('_')[0], current_time.split('_')[1], identity, access_status])
                    last_denied_timestamp[identity] = datetime.strptime(current_time, "%d-%m-%Y_%H:%M:%S")
        else:
            cv2.putText(gbr1, 'Unknown', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
            db.reference('test').update({'string': " ", 'int': 3})
    else:
        cv2.putText(gbr1, 'Unknown', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
        db.reference('test').update({'string': "Unknown", 'int': 3})

    cv2.rectangle(gbr1, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow('res', gbr1)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()