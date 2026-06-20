import cv2
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.models import load_model
import joblib
from collections import deque


import warnings
warnings.filterwarnings('ignore')

from mediapipe.python.solutions import holistic as mp_holistic
from mediapipe.python.solutions import drawing_utils as mp_drawing
import mediapipe.python.solutions.hands as mp_hands

# ============================================================
# UCITAJ MODEL I POMOCNE FAJLOVE
# ============================================================
model   = load_model("models/best_model.keras")
scaler  = joblib.load("models/scaler.pkl")
classes = np.load("models/classes.npy", allow_pickle=True)

FRAME_SIZE  = 33*4 + 21*3 + 21*3  # 258
NUM_FRAMES  = 40
CONFIDENCE  = 0.4  # minimalna sigurnost za prikaz


# Buffer koji cuva poslednjih 40 frejmova
buffer = deque(maxlen=NUM_FRAMES)

# Za formiranje reci
current_word   = ""
last_letter    = ""
stable_counter = 0
STABLE_FRAMES  = 15  # koliko frejmova isto slovo mora biti prepoznato


IDLE = "cekanje"      # ruka nije u sceni
RECORDING = "snimam"  # ruka ušla, snimam 40 frejmova
PREDICTING = "predikujem"  # skupio 40 frejmova, predikujem


state = IDLE
record_buffer = [] 
predicted_letter = ""
confidence_val   = 0.0
current_word     = ""

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

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)



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

         # Stilovi za crtanje keypointa
        # Desna ruka - zelena
        right_hand_style = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=2)
        right_conn_style  = mp_drawing.DrawingSpec(color=(0, 200, 0), thickness=1)

        # Leva ruka - plava
        left_hand_style  = mp_drawing.DrawingSpec(color=(255, 100, 0), thickness=1, circle_radius=2)
        left_conn_style   = mp_drawing.DrawingSpec(color=(200, 80,  0), thickness=1)

        # Telo (pose) - bela tacka, siva linija; pose sadrzi i face tacke (0-10),
        # pa crtamo samo tacke tela (11+) rucno kako bismo preskocili lice
        pose_dot_style  = mp_drawing.DrawingSpec(color=(200, 200, 200), thickness=1, circle_radius=2)
        pose_conn_style = mp_drawing.DrawingSpec(color=(120, 120, 120), thickness=1)

        # Crtamo pose (telo + lice) - lice je zuto, telo sivo
        if results.pose_landmarks:
            h_img, w_img = image.shape[:2]
            face_indices = set(range(0, 11))  # 0-10 su lice/glava u Holistic pose

            # Prvo nacrtaj telo standardnom funkcijom
            # Privremeno sakrij face tacke
            orig_vis = {}
            for idx in face_indices:
                lm = results.pose_landmarks.landmark[idx]
                orig_vis[idx] = lm.visibility
                lm.visibility = 0.0

            mp_drawing.draw_landmarks(
                image, results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=pose_dot_style,
                connection_drawing_spec=pose_conn_style
            )

            # Vrati i rucno nacrtaj face tacke zucom bojom, malim kruzicima
            for idx in face_indices:
                results.pose_landmarks.landmark[idx].visibility = orig_vis[idx]

            for idx in face_indices:
                lm = results.pose_landmarks.landmark[idx]
                if lm.visibility > 0.3:
                    cx = int(lm.x * w_img)
                    cy = int(lm.y * h_img)
                    cv2.circle(image, (cx, cy), 2, (0, 220, 255), -1)  # zuto-narandzasta

        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            landmark_drawing_spec=right_hand_style,
            connection_drawing_spec=right_conn_style
        )
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            landmark_drawing_spec=left_hand_style,
            connection_drawing_spec=left_conn_style
        )

        # Izvuci keypoints i dodaj u buffer
        keypoints = extract_keypoints(results)
        # buffer.append(keypoints)

        hand_detected = results.left_hand_landmarks or results.right_hand_landmarks

        # # Stilovi za crtanje keypointa
        # # Desna ruka - zelena
        # right_hand_style = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=2)
        # right_conn_style  = mp_drawing.DrawingSpec(color=(0, 200, 0), thickness=1)

        # # Leva ruka - plava
        # left_hand_style  = mp_drawing.DrawingSpec(color=(255, 100, 0), thickness=1, circle_radius=2)
        # left_conn_style   = mp_drawing.DrawingSpec(color=(200, 80,  0), thickness=1)

        # # Telo (pose) - bela tacka, siva linija; pose sadrzi i face tacke (0-10),
        # # pa crtamo samo tacke tela (11+) rucno kako bismo preskocili lice
        # pose_dot_style  = mp_drawing.DrawingSpec(color=(200, 200, 200), thickness=1, circle_radius=2)
        # pose_conn_style = mp_drawing.DrawingSpec(color=(120, 120, 120), thickness=1)

        # # Crtaj keypoints na slici
        # mp_drawing.draw_landmarks(image, results.right_hand_landmarks,
        #                           mp_hands.HAND_CONNECTIONS)
        # mp_drawing.draw_landmarks(image, results.left_hand_landmarks,
        #                           mp_hands.HAND_CONNECTIONS)
        # mp_drawing.draw_landmarks(image, results.pose_landmarks,
        #                   mp_holistic.POSE_CONNECTIONS)

        # # Izvuci keypoints i dodaj u buffer
        # keypoints = extract_keypoints(results)
        # # buffer.append(keypoints)

        # hand_detected = results.left_hand_landmarks or results.right_hand_landmarks

        # if not hand_detected:
        #     predicted_letter = ""
        #     confidence_val = 0.0
        #     buffer.clear()

        # predicted_letter = ""
        # confidence_val   = 0.0

        # # Kada imamo 40 frejmova - predikuj
        # if len(buffer) == NUM_FRAMES:
        #     # Pripremi ulaz za model
        #     X = np.array(buffer)                          # (40, 258)
        #     X_2d = X.reshape(1, NUM_FRAMES * FRAME_SIZE)  # (1, 10320)
        #     X_scaled = scaler.transform(X_2d)             # normalizacija
        #     X_3d = X_scaled.reshape(1, NUM_FRAMES, FRAME_SIZE)  # (1, 40, 258)

        #     # Predikuj
        #     prediction  = model.predict(X_3d, verbose=0)
        #     class_idx   = np.argmax(prediction)
        #     confidence_val = prediction[0][class_idx]

        #     if confidence_val >= CONFIDENCE:
        #         predicted_letter = classes[class_idx]

        #         # Formiranje reci - dodaj slovo samo ako je stabilno
        #         if predicted_letter == last_letter:
        #             stable_counter += 1
        #         else:
        #             stable_counter = 0
        #             last_letter = predicted_letter

                # if stable_counter == STABLE_FRAMES:
                #     current_word += predicted_letter
                #     stable_counter = 0

        if state == IDLE:
            if hand_detected:
                state = RECORDING
                record_buffer = [keypoints]
                predicted_letter = ""
                confidence_val = 0.0

        elif state == RECORDING:
            record_buffer.append(keypoints)

            if not hand_detected:
                # Ruka izasla pre 40 frejmova - resetuj
                state = IDLE
                record_buffer = []
            elif len(record_buffer) == NUM_FRAMES:
                # Skupili 40 frejmova - predikuj
                state = PREDICTING

        elif state == PREDICTING:
            # Predikuj
            X = np.array(record_buffer)
            X_2d = X.reshape(1, NUM_FRAMES * FRAME_SIZE)
            X_scaled = scaler.transform(X_2d)
            X_3d = X_scaled.reshape(1, NUM_FRAMES, FRAME_SIZE)

            prediction = model.predict(X_3d, verbose=0)
            class_idx = np.argmax(prediction)
            confidence_val = prediction[0][class_idx]

            if confidence_val >= CONFIDENCE:
                predicted_letter = classes[class_idx]
            else:
                predicted_letter = "?"

            # Cekaj da ruka izadje pa ponovi
            if not hand_detected:
                state = IDLE
                record_buffer = []

        # ============================================================
        # PRIKAZ NA EKRANU
        # ============================================================
        # h, w = image.shape[:2]

        # # Pozadina za tekst
        # cv2.rectangle(image, (0, h-120), (w, h), (0,0,0), -1)

        # # Trenutno prepoznato slovo
        # cv2.putText(image, f"Slovo: {predicted_letter}",
        #             (10, h-80), cv2.FONT_HERSHEY_SIMPLEX,
        #             1.2, (0,255,0), 2)

        # # Sigurnost
        # cv2.putText(image, f"Sigurnost: {confidence_val*100:.0f}%",
        #             (10, h-50), cv2.FONT_HERSHEY_SIMPLEX,
        #             0.7, (255,255,0), 2)

        # # Formirana rec
        # cv2.putText(image, f"Rec: {current_word}",
        #             (10, h-15), cv2.FONT_HERSHEY_SIMPLEX,
        #             1.0, (255,255,255), 2)

        # # Uputstvo
        # cv2.putText(image, "SPACE=obrisi rec  ESC=izlaz",
        #             (w-320, 25), cv2.FONT_HERSHEY_SIMPLEX,
        #             0.6, (200,200,200), 1)

        # # Broj frejmova u bufferu
        # cv2.putText(image, f"Buffer: {len(buffer)}/{NUM_FRAMES}",
        #             (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
        #             0.6, (200,200,200), 1)

        # cv2.imshow("Srpski znakovni jezik", image)

        # # Tasteri
        # key = cv2.waitKey(1) & 0xFF
        # if key == 27:    # ESC - izlaz
        #     break
        # elif key == 32:  # SPACE - obrisi rec
        #     current_word = ""
        # elif key == 13:    # ENTER - potvrdi trenutno slovo odmah
        #     if predicted_letter:
        #         current_word += predicted_letter
        #         stable_counter = 0
        # elif key == 8:     # BACKSPACE - obrisi zadnje slovo
        #     current_word = current_word[:-1]

        h, w = image.shape[:2]

        cv2.rectangle(image, (0, h-130), (w, h), (0,0,0), -1)

        cv2.putText(image, f"Stanje: {state}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

        if state == RECORDING:
            cv2.putText(image, f"Frejmovi: {len(record_buffer)}/{NUM_FRAMES}",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)

        cv2.putText(image, f"Slovo: {predicted_letter}",
                    (10, h-90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)

        cv2.putText(image, f"Sigurnost: {confidence_val*100:.0f}%",
                    (10, h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

        cv2.putText(image, f"Rec: {current_word}",
                    (10, h-25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

        cv2.putText(image, "ENTER=potvrdi  BACKSPACE=obrisi  SPACE=reset  ESC=izlaz",
                    (10, h-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150,150,150), 1)

        cv2.imshow("Srpski znakovni jezik", image)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:   # ESC
            break
        elif key == 32: # SPACE - obrisi rec
            current_word = ""
        elif key == 13: # ENTER - potvrdi slovo
            if predicted_letter and predicted_letter != "?":
                current_word += predicted_letter
        elif key == 8:  # BACKSPACE
            current_word = current_word[:-1]
cap.release()
cv2.destroyAllWindows()