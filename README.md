# Facility-Access-System-using-Face-Recognition

# Overview

The Facility Access System is designed to enhance security by using face recognition technology to control access to restricted areas. This system detects faces in real-time and grants or denies access based on a pre-registered database of authorized individuals. The system integrates various technologies including OpenCV for face detection, FaceNet for face recognition, and Firebase and Google Sheets for logging access attempts.

# Pre-processing

**Face Detection and Cropping:** Faces are detected in images using **OpenCV**'s Haar Cascade classifier. Detected faces are then cropped and saved for further processing.

**Folder Management:** The system organizes images into directories and processes them to create a face database. Unnecessary directories are removed after processing.

# Face Embedding and Database Management

**Face Embeddings:** Faces are resized and converted to embeddings using the **FaceNet** model. These embeddings are stored in a dictionary for recognition.

**Database Handling:** The system maintains a pickled file (data.pkl) to store and retrieve face embeddings for registered individuals.

# Access Control Logic

**Real-Time Face Recognition:** The system continuously captures video frames and detects faces.

**Hand Gesture Recognition:** Uses **MediaPipe** to recognize specific hand gestures as an additional security layer.

**Access Decision:** Compares detected face embeddings against the stored database

**Access Granted:** If a face is recognized and authorized, access is granted.

**Access Denied:** If a face is recognized but not authorized, access is denied.

**Unknown:** If no face match is found, the person is marked as unknown.

# Logging and Notification

**CSV Logging:** Logs each access attempt with date, time, name, and access status.

**Google Sheets Logging:** Logs access attempts to a Google Sheet for easy monitoring.

**Firebase Updates:** Updates a Firebase real-time database with access attempt information for potential integration with other systems.

