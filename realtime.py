import cv2
import numpy as np
import os

os.environ["KERAS_BACKEND"] = "tensorflow"

from keras.models import load_model
import joblib
from collections import deque


from mediapipe.python.solutions import holistic as mp_holistic
from mediapipe.python.solutions import drawing_utils as mp_drawing


# ============================================================
# UCITAJ MODEL I POMOCNE FAJLOVE
# ============================================================
model   = load_model("models/best_model.keras")
scaler  = joblib.load("models/scaler.pkl")
classes = np.load("models/classes.npy", allow_pickle=True)

FRAME_SIZE  = 33*4 + 21*3 + 21*3  # 258
NUM_FRAMES  = 40
CONFIDENCE  = 0.7  # minimalna sigurnost za prikaz

# Buffer koji cuva poslednjih 40 frejmova
buffer = deque(maxlen=NUM_FRAMES)

# Za formiranje reci
current_word   = ""
last_letter    = ""
stable_counter = 0
STABLE_FRAMES  = 15  # koliko frejmova isto slovo mora biti prepoznato

# ============================================================
# MEDIAPIPE SETUP
# ============================================================


def extract_keypoints(results):
    """Izvlaci keypoints iz MediaPipe rezultata - isto kao u datasetu."""
    
    # Pose (telo) - 33 tacke x 4 vrednosti
    if results.pose_landmarks:
        pose = np.array([[lm.x, lm.y, lm.z, lm.visibility] 
                         for lm in results.pose_landmarks.landmark]).flatten()
    else:
        pose = np.zeros(33 * 4)
    
    # Leva ruka - 21 tacka x 3 vrednosti
    if results.left_hand_landmarks:
        left = np.array([[lm.x, lm.y, lm.z] 
                         for lm in results.left_hand_landmarks.landmark]).flatten()
    else:
        left = np.zeros(21 * 3)
    
    # Desna ruka - 21 tacka x 3 vrednosti
    if results.right_hand_landmarks:
        right = np.array([[lm.x, lm.y, lm.z] 
                          for lm in results.right_hand_landmarks.landmark]).flatten()
    else:
        right = np.zeros(21 * 3)
    
    return np.concatenate([pose, left, right])  # (258,)


# ============================================================
# GLAVNA PETLJA
# ============================================================
cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5,
                           min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # MediaPipe obrada
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Crtaj keypoints na slici
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks,
                                  mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks,
                                  mp_holistic.HAND_CONNECTIONS)

        # Izvuci keypoints i dodaj u buffer
        keypoints = extract_keypoints(results)
        buffer.append(keypoints)

        predicted_letter = ""
        confidence_val   = 0.0

        # Kada imamo 40 frejmova - predikuj
        if len(buffer) == NUM_FRAMES:
            # Pripremi ulaz za model
            X = np.array(buffer)                          # (40, 258)
            X_2d = X.reshape(1, NUM_FRAMES * FRAME_SIZE)  # (1, 10320)
            X_scaled = scaler.transform(X_2d)             # normalizacija
            X_3d = X_scaled.reshape(1, NUM_FRAMES, FRAME_SIZE)  # (1, 40, 258)

            # Predikuj
            prediction  = model.predict(X_3d, verbose=0)
            class_idx   = np.argmax(prediction)
            confidence_val = prediction[0][class_idx]

            if confidence_val >= CONFIDENCE:
                predicted_letter = classes[class_idx]

                # Formiranje reci - dodaj slovo samo ako je stabilno
                if predicted_letter == last_letter:
                    stable_counter += 1
                else:
                    stable_counter = 0
                    last_letter = predicted_letter

                if stable_counter == STABLE_FRAMES:
                    current_word += predicted_letter
                    stable_counter = 0

        # ============================================================
        # PRIKAZ NA EKRANU
        # ============================================================
        h, w = image.shape[:2]

        # Pozadina za tekst
        cv2.rectangle(image, (0, h-120), (w, h), (0,0,0), -1)

        # Trenutno prepoznato slovo
        cv2.putText(image, f"Slovo: {predicted_letter}",
                    (10, h-80), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0,255,0), 2)

        # Sigurnost
        cv2.putText(image, f"Sigurnost: {confidence_val*100:.0f}%",
                    (10, h-50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255,255,0), 2)

        # Formirana rec
        cv2.putText(image, f"Rec: {current_word}",
                    (10, h-15), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (255,255,255), 2)

        # Uputstvo
        cv2.putText(image, "SPACE=obrisi rec  ESC=izlaz",
                    (w-320, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (200,200,200), 1)

        # Broj frejmova u bufferu
        cv2.putText(image, f"Buffer: {len(buffer)}/{NUM_FRAMES}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (200,200,200), 1)

        cv2.imshow("Srpski znakovni jezik", image)

        # Tasteri
        key = cv2.waitKey(1) & 0xFF
        if key == 27:    # ESC - izlaz
            break
        elif key == 32:  # SPACE - obrisi rec
            current_word = ""

cap.release()
cv2.destroyAllWindows()