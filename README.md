The Facial Recognition project is my first venture into the world of computer vision and artificial intelligence. 
This project focuses on developing a system capable of detecting and recognizing human faces in real-time using a webcam. 
The core objective is to build a reliable and efficient facial recognition tool that can identify individuals based on their unique facial features.

The project uses modern face detection and recognition techniques, 
integrating computer vision libraries such as OpenCV and face recognition algorithms. 
It demonstrates key concepts like facial landmark detection, face encoding, image preprocessing, and recognition accuracy handling.

Key Features:

Real-time face detection and recognition
Accurate matching with stored facial data
User-friendly graphical interface (in progress)
Modular design for future upgrades, such as emotion detection or multi-face support

Technology Stack:
Python
OpenCV
Dlib / Mediapipe / Face Recognition (based on implementation)
PyQt6 (for GUI, in progress)

1. **Install Python 3.10**
   Download and install Python version 3.10 from the official [Python website](https://www.python.org/downloads/).

2. **Install Required Python Modules**
   Use `pip` to install the following modules:

   ```bash
   pip install opencv-python dlib numpy pickle-mixin pyttsx3
   ```

3. **Download Pre-trained Face Recognition Model**
   Ensure that the file named **`dlib_face_recognition_resnet_model_v1.dat`** is present in the project directory.
   You can download it from the official dlib model repository:
   [http://dlib.net/files/dlib\_face\_recognition\_resnet\_model\_v1.dat.bz2](http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2)

   or
  **File is already given**
4. The last file can be downloaded instantly. ***Press download button right-upper side***(https://github.com/ageitgey/face_recognition_models/blob/master/face_recognition_models/models/shape_predictor_68_face_landmarks.dat)
