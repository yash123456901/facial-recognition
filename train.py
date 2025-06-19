import os
import cv2
import dlib
import pickle

# Paths
KNOWN_FACES_DIR = "known_faces"
ENCODINGS_FILE = "encodings.pickle"
SHAPE_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
FACE_RECOGNITION_MODEL_PATH = "dlib_face_recognition_resnet_model_v1.dat"

# Load dlib models
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
face_encoder = dlib.face_recognition_model_v1(FACE_RECOGNITION_MODEL_PATH)

# Data to save
known_encodings = []
known_names = []

# Loop over known face images
for filename in os.listdir(KNOWN_FACES_DIR):
    if not filename.lower().endswith(".jpg"):
        continue

    name = os.path.splitext(filename)[0]
    path = os.path.join(KNOWN_FACES_DIR, filename)

    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_detector(gray)
    if not faces:
        print(f"[WARN] No face in {filename}, skipping.")
        continue

    for face in faces:
        shape = shape_predictor(image, face)
        encoding = face_encoder.compute_face_descriptor(image, shape)
        known_encodings.append(list(encoding))  # convert dlib vector to list
        known_names.append(name)

# Save encodings
with open(ENCODINGS_FILE, "wb") as f:
    pickle.dump({"encodings": known_encodings, "names": known_names}, f)

print(f"[INFO] Saved {len(known_encodings)} face encoding(s).")
