import cv2
import dlib
import numpy as np
import pickle
import pyttsx3

# Load encodings
with open("encodings.pickle", "rb") as f:
    data = pickle.load(f)

# Load models
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_encoder = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Greeted persons cache
greeted = set()

def recognize_face(encoding):
    matches = []
    for known_encoding in data["encodings"]:
        dist = np.linalg.norm(np.array(known_encoding) - np.array(encoding))
        matches.append(dist)
    min_dist = min(matches)
    if min_dist < 0.6:  # Threshold
        return data["names"][matches.index(min_dist)]
    return "Unknown"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        shape = shape_predictor(frame, face)

        # Draw white dot landmarks
        for i in range(68):
            part = shape.part(i)
            cv2.circle(frame, (part.x, part.y), 2, (255, 255, 255), -1)

        # Encode face
        face_descriptor = face_encoder.compute_face_descriptor(frame, shape)
        encoding = np.array(face_descriptor)

        name = recognize_face(encoding)

        # Draw name and rectangle
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 255, 200), 1)
        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 255, 200), 2)

        # Greet once
        if name != "Unknown" and name not in greeted:
            greeted.add(name)
            engine.say(f"Hello, {name}")
            engine.runAndWait()

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()