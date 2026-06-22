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
from interpolation import interpolate_sequence_gaps
from body_normalization import normalize_sequence
from curl_smoothing import smooth_curl_sequence

from mediapipe.python.solutions import holistic as mp_holistic
from mediapipe.python.solutions import drawing_utils as mp_drawing
import mediapipe.python.solutions.hands as mp_hands

model   = load_model("models/best_model.keras")
scaler  = joblib.load("models/scaler.pkl")
classes = np.load("models/classes.npy", allow_pickle=True)

WRIST = 0
THUMB_TIP, THUMB_MCP = 4, 2
INDEX_TIP, INDEX_MCP = 8, 5
MIDDLE_TIP, MIDDLE_MCP = 12, 9
RING_TIP, RING_MCP = 16, 13
PINKY_TIP, PINKY_MCP = 20, 17

FINGER_TIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
FINGER_MCPS = [THUMB_MCP, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]

POSE_LEN = 33 * 4
HAND_LEN = 21 * 3
LEFT_HAND_START = POSE_LEN
RIGHT_HAND_START = POSE_LEN + HAND_LEN


def get_hand_landmarks(frame, hand_start):
    hand_flat = frame[hand_start: hand_start + HAND_LEN]
    return hand_flat.reshape(21, 3)


def finger_curl(landmarks, tip_idx, mcp_idx, wrist_idx=WRIST):
    wrist = landmarks[wrist_idx]
    tip = landmarks[tip_idx]
    mcp = landmarks[mcp_idx]

    if np.all(landmarks == 0):
        return 0.0

    dist_tip_wrist = np.linalg.norm(tip - wrist)
    dist_mcp_wrist = np.linalg.norm(mcp - wrist) + 1e-6

    return dist_tip_wrist / dist_mcp_wrist


def compute_curl_features(frame):
    left_hand = get_hand_landmarks(frame, LEFT_HAND_START)
    right_hand = get_hand_landmarks(frame, RIGHT_HAND_START)

    curls = []
    for tip, mcp in zip(FINGER_TIPS, FINGER_MCPS):
        curls.append(finger_curl(left_hand, tip, mcp))
    for tip, mcp in zip(FINGER_TIPS, FINGER_MCPS):
        curls.append(finger_curl(right_hand, tip, mcp))

    return np.array(curls, dtype=np.float64)  # (10,)


def add_curl_features_to_sequence(sequence):
    curl_features = np.array([compute_curl_features(frame) for frame in sequence])
    curl_features = smooth_curl_sequence(curl_features, window=3)
    return np.concatenate([sequence, curl_features], axis=1)


FRAME_SIZE = 33*4 + 21*3 + 21*3 
NUM_FRAMES = 40
CONFIDENCE = 0.4

def extract_keypoints(results):
    if results.pose_landmarks:
        pose = np.array([[lm.x, lm.y, lm.z, lm.visibility]
                         for lm in results.pose_landmarks.landmark]).flatten()
    else:
        pose = np.zeros(33 * 4)

    if results.left_hand_landmarks:
        left = np.array([[lm.x, lm.y, lm.z]
                         for lm in results.left_hand_landmarks.landmark]).flatten()
    else:
        left = np.zeros(21 * 3)

    if results.right_hand_landmarks:
        right = np.array([[lm.x, lm.y, lm.z]
                          for lm in results.right_hand_landmarks.landmark]).flatten()
    else:
        right = np.zeros(21 * 3)

    return np.concatenate([pose, left, right])

IDLE       = "waiting"
RECORDING  = "recording"
PREDICTING = "predicting"

state            = IDLE
record_buffer    = []
predicted_letter = ""
confidence_val   = 0.0
current_word     = ""
top5             = []  

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

with mp_holistic.Holistic(min_detection_confidence=0.5,
                           min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Crtanje keypointa
        right_hand_style = mp_drawing.DrawingSpec(color=(0, 255, 0),   thickness=1, circle_radius=2)
        right_conn_style  = mp_drawing.DrawingSpec(color=(0, 200, 0),   thickness=1)
        left_hand_style  = mp_drawing.DrawingSpec(color=(255, 100, 0),  thickness=1, circle_radius=2)
        left_conn_style   = mp_drawing.DrawingSpec(color=(200, 80,  0),  thickness=1)
        pose_dot_style   = mp_drawing.DrawingSpec(color=(200, 200, 200), thickness=1, circle_radius=2)
        pose_conn_style  = mp_drawing.DrawingSpec(color=(120, 120, 120), thickness=1)

        if results.pose_landmarks:
            face_indices = set(range(0, 11))
            orig_vis = {}
            for idx in face_indices:
                lm = results.pose_landmarks.landmark[idx]
                orig_vis[idx] = lm.visibility
                lm.visibility = 0.0

            mp_drawing.draw_landmarks(image, results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=pose_dot_style,
                connection_drawing_spec=pose_conn_style)

            for idx in face_indices:
                results.pose_landmarks.landmark[idx].visibility = orig_vis[idx]

            h_img, w_img = image.shape[:2]
            for idx in face_indices:
                lm = results.pose_landmarks.landmark[idx]
                if lm.visibility > 0.3:
                    cx = int(lm.x * w_img)
                    cy = int(lm.y * h_img)
                    cv2.circle(image, (cx, cy), 2, (0, 220, 255), -1)

        mp_drawing.draw_landmarks(image, results.right_hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            landmark_drawing_spec=right_hand_style,
            connection_drawing_spec=right_conn_style)
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            landmark_drawing_spec=left_hand_style,
            connection_drawing_spec=left_conn_style)

        keypoints    = extract_keypoints(results)
        hand_detected = results.left_hand_landmarks or results.right_hand_landmarks

        # State machine
        if state == IDLE:
            if hand_detected:
                state         = RECORDING
                record_buffer = [keypoints]
                predicted_letter = ""
                confidence_val   = 0.0
                top5             = []

        elif state == RECORDING:
            record_buffer.append(keypoints)
            if not hand_detected:
                state         = IDLE
                record_buffer = []
            elif len(record_buffer) == NUM_FRAMES:
                state = PREDICTING

        elif state == PREDICTING:
            X_raw = np.array(record_buffer)                          # (40, 258)
            X_raw, n_left_filled, n_right_filled = interpolate_sequence_gaps(X_raw)
            if n_left_filled > 0 or n_right_filled > 0:
                print(f"Popunjeno rupa - leva ruka: {n_left_filled}, desna ruka: {n_right_filled}")

            from body_normalization import _compute_center_and_scale
            scales = []
            for frame in X_raw:
                cs = _compute_center_and_scale(frame)
                if cs is not None:
                    scales.append(cs[2])
            if scales:
                print(f"\n--- DIAGNOSTICS  ---")
                print(f"Scale (shoulder distance) - min: {min(scales):.4f}, max: {max(scales):.4f}, prosek: {np.mean(scales):.4f}")
                print(f"Number of frames without detected shoulders: {NUM_FRAMES - len(scales)}/{NUM_FRAMES}")
            else:
                print(f"\n--- DIAGNOSTICS: NO FRAMES WITH DETECTED SHOULDERS! ---")

            X_norm_body = normalize_sequence(X_raw)                      
            X_with_curl = add_curl_features_to_sequence(X_norm_body)     

            if scales:
                curl_part = X_with_curl[:, 258:268]  
                print(f"Curl features (average across 40 frames): {curl_part.mean(axis=0).round(2)}")
                print(f"--- END OF DIAGNOSTICS ---\n")

            X_2d   = X_with_curl.reshape(1, NUM_FRAMES * X_with_curl.shape[1])
            X_sc   = scaler.transform(X_2d)
            X_3d   = X_sc.reshape(1, NUM_FRAMES, X_with_curl.shape[1])

            probs  = model.predict(X_3d, verbose=0)[0]

            # Top 5 
            top_idx = np.argsort(probs)[::-1][:5]
            top5    = [(classes[i], probs[i]) for i in top_idx]

            class_idx      = top_idx[0]
            confidence_val = probs[class_idx]
            predicted_letter = classes[class_idx] if confidence_val >= CONFIDENCE else "?"

            if not hand_detected:
                state         = IDLE
                record_buffer = []

        # ============================================================
        # UI
        # ============================================================
        h, w = image.shape[:2]

        # Left side
        cv2.rectangle(image, (0, h - 130), (w // 2, h), (0, 0, 0), -1)

        state_color = (0, 255, 255) if state == RECORDING else (200, 200, 200)
        cv2.putText(image, f"Stanje: {state}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 1)

        if state == RECORDING:
            # Progress bar
            progress = int((len(record_buffer) / NUM_FRAMES) * 300)
            cv2.rectangle(image, (10, 50), (310, 65), (50, 50, 50), -1)
            cv2.rectangle(image, (10, 50), (10 + progress, 65), (0, 255, 255), -1)
            cv2.putText(image, f"{len(record_buffer)}/{NUM_FRAMES}",
                        (315, 63), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.putText(image, f"Letter: {predicted_letter}",
                    (10, h - 90), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 2)

        # Confidence bar for main letter
        if confidence_val > 0:
            bar_w = int(confidence_val * 200)
            bar_color = (0, 255, 0) if confidence_val >= 0.7 else \
                        (0, 165, 255) if confidence_val >= 0.4 else (0, 0, 255)
            cv2.rectangle(image, (10, h - 75), (210, h - 60), (50, 50, 50), -1)
            cv2.rectangle(image, (10, h - 75), (10 + bar_w, h - 60), bar_color, -1)
            cv2.putText(image, f"{confidence_val * 100:.0f}%",
                        (215, h - 62), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.putText(image, f"Word: {current_word}",
                    (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        cv2.putText(image, "ENTER=confirm  BACKSPACE=delete  SPACE=reset  ESC=exit",
                    (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        # Right side - TOP 5 debug panel
        panel_x = w - 280
        cv2.rectangle(image, (panel_x - 10, 0), (w, 200), (20, 20, 20), -1)
        cv2.putText(image, "TOP 5:", (panel_x, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

        for rank, (lbl, prob) in enumerate(top5):
            y_pos    = 55 + rank * 30
            bar_len  = int(prob * 200)
            color    = (0, 255, 0) if rank == 0 else (100, 100, 200)
            cv2.rectangle(image, (panel_x, y_pos - 12), (panel_x + bar_len, y_pos + 3), color, -1)
            cv2.putText(image, f"{lbl}: {prob*100:.1f}%",
                        (panel_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1)

        cv2.imshow("Serbian sign language - DEBUG", image)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == 32:
            current_word = ""
        elif key == 13:
            if predicted_letter and predicted_letter != "?":
                current_word += predicted_letter
        elif key == 8:
            current_word = current_word[:-1]

cap.release()
cv2.destroyAllWindows()