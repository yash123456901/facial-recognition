import cv2
import os

KNOWN_FACES_DIR = "known_faces"

if not os.path.exists(KNOWN_FACES_DIR):
    os.makedirs(KNOWN_FACES_DIR)

name = input("Enter your name: ").strip()
print("[INFO] Press 's' to capture face, or 'q' to quit.")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    cv2.imshow("Capture Face", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        cv2.imwrite(os.path.join(KNOWN_FACES_DIR, f"{name}.jpg"), frame)
        print("[INFO] Face saved.")
        break
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
