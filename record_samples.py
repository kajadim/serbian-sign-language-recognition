"""
Snimanje novih primera za dataset - identican format originalnom datasetu.
Cuva .npy fajlove u data/<SLOVO>/ folder, gotove za load_and_explore_data.py

KORISCENJE:
    python record_samples.py D
    python record_samples.py ZH
    python record_samples.py V

Kontrole:
    SPACE - pokreni/zaustavi rucno (opciono, inace je automatski)
    ESC   - izlaz
"""
import cv2
import numpy as np
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from mediapipe.python.solutions import holistic as mp_holistic
from mediapipe.python.solutions import drawing_utils as mp_drawing
import mediapipe.python.solutions.hands as mp_hands

# ============================================================
# PODESAVANJA - ista kao u datasetu (FRAME_SIZE=258, NUM_FRAMES=40)
# ============================================================
FRAME_SIZE = 33 * 4 + 21 * 3 + 21 * 3  # 258
NUM_FRAMES = 40
DATA_PATH  = "data"
OUT_OF_SCENE_GRACE = 5  # frejmova tolerancije pre nego sto se ruka smatra "izasla"


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


def main():
    if len(sys.argv) < 2:
        print("Koriscenje: python record_samples.py <SLOVO>")
        print("Primer: python record_samples.py D")
        sys.exit(1)

    letter = sys.argv[1].upper()
    out_dir = os.path.join(DATA_PATH, letter)
    os.makedirs(out_dir, exist_ok=True)

    existing = [f for f in os.listdir(out_dir) if f.endswith('.npy')]
    print(f"Slovo: {letter}")
    print(f"Trenutno u datasetu: {len(existing)} snimaka")
    print(f"Novi snimci ce se dodati u: {out_dir}")
    print("\nUputstvo:")
    print("  - Drzite ruke van scene dok ne pocnete znak")
    print("  - Cim ruka udje u kadar, snimanje automatski pocinje")
    print("  - Snima se tacno 40 frejmova")
    print("  - Izvedite znak prirodno, ruka ulazi - znak - ruka izlazi")
    print("  - ESC za izlaz\n")

    IDLE = "cekanje"
    RECORDING = "snimam"
    SAVED = "sacuvano"

    state = IDLE
    record_buffer = []
    out_of_scene_count = 0
    saved_count = 0
    save_flash_timer = 0

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

            # Crtanje (zelena=desna, plava=leva, siva=telo, zuta=lice)
            right_style = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=2)
            right_conn  = mp_drawing.DrawingSpec(color=(0, 200, 0), thickness=1)
            left_style  = mp_drawing.DrawingSpec(color=(255, 100, 0), thickness=1, circle_radius=2)
            left_conn   = mp_drawing.DrawingSpec(color=(200, 80, 0), thickness=1)
            pose_style  = mp_drawing.DrawingSpec(color=(200, 200, 200), thickness=1, circle_radius=2)
            pose_conn   = mp_drawing.DrawingSpec(color=(120, 120, 120), thickness=1)

            if results.pose_landmarks:
                face_idx = set(range(0, 11))
                orig_vis = {}
                for idx in face_idx:
                    lm = results.pose_landmarks.landmark[idx]
                    orig_vis[idx] = lm.visibility
                    lm.visibility = 0.0
                mp_drawing.draw_landmarks(image, results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=pose_style, connection_drawing_spec=pose_conn)
                for idx in face_idx:
                    results.pose_landmarks.landmark[idx].visibility = orig_vis[idx]

                h_img, w_img = image.shape[:2]
                for idx in face_idx:
                    lm = results.pose_landmarks.landmark[idx]
                    if lm.visibility > 0.3:
                        cx, cy = int(lm.x * w_img), int(lm.y * h_img)
                        cv2.circle(image, (cx, cy), 2, (0, 220, 255), -1)

            mp_drawing.draw_landmarks(image, results.right_hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=right_style, connection_drawing_spec=right_conn)
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=left_style, connection_drawing_spec=left_conn)

            keypoints = extract_keypoints(results)
            hand_detected = bool(results.left_hand_landmarks or results.right_hand_landmarks)

            # ============================================================
            # STATE MACHINE - identicna logika originalnom datasetu:
            # ruka udje -> snimaj 40 frejmova -> sacuvaj
            # 5 frejmova tolerancije ako ruka nakratko "trepne" iz detekcije
            # ============================================================
            if state == IDLE:
                if hand_detected:
                    state = RECORDING
                    record_buffer = [keypoints]
                    out_of_scene_count = 0

            elif state == RECORDING:
                if hand_detected:
                    record_buffer.append(keypoints)
                    out_of_scene_count = 0
                else:
                    out_of_scene_count += 1
                    record_buffer.append(keypoints)  # i dalje dodaj (nule)

                if len(record_buffer) >= NUM_FRAMES:
                    # Sacuvaj tacno 40 frejmova
                    data_to_save = np.array(record_buffer[:NUM_FRAMES], dtype=np.float64)
                    timestamp = int(time.time() * 1000)
                    filename = f"{timestamp}-MANUAL.npy"
                    filepath = os.path.join(out_dir, filename)
                    np.save(filepath, data_to_save)
                    saved_count += 1
                    save_flash_timer = 15
                    print(f"Sacuvano: {filename}  (ukupno novih: {saved_count})")

                    state = IDLE
                    record_buffer = []
                    out_of_scene_count = 0

                elif out_of_scene_count >= OUT_OF_SCENE_GRACE:
                    # Ruka izasla prerano - odbaci
                    print("Ruka izasla prerano, snimak odbacen.")
                    state = IDLE
                    record_buffer = []
                    out_of_scene_count = 0

            # ============================================================
            # UI
            # ============================================================
            h, w = image.shape[:2]
            cv2.rectangle(image, (0, 0), (w, 90), (0, 0, 0), -1)

            cv2.putText(image, f"Slovo: {letter}   |   Novih snimaka: {saved_count}   |   Ukupno u datasetu: {len(existing) + saved_count}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            state_color = (0, 255, 255) if state == RECORDING else (200, 200, 200)
            cv2.putText(image, f"Stanje: {state}",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 1)

            if state == RECORDING:
                progress = int((len(record_buffer) / NUM_FRAMES) * 400)
                cv2.rectangle(image, (10, 65), (410, 80), (50, 50, 50), -1)
                cv2.rectangle(image, (10, 65), (10 + progress, 80), (0, 255, 255), -1)
                cv2.putText(image, f"{len(record_buffer)}/{NUM_FRAMES}",
                            (420, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            if save_flash_timer > 0:
                cv2.putText(image, "SACUVANO!", (w // 2 - 100, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                save_flash_timer -= 1

            cv2.putText(image, "ESC = izlaz",
                        (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

            cv2.imshow(f"Snimanje slova {letter}", image)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nZavrseno! Snimljeno {saved_count} novih primera za slovo '{letter}'.")
    print(f"Fajlovi su u: {out_dir}")
    print("\nSledeci koraci:")
    print("  1. python load_and_explore_data.py   (proveri da se novi snimci ucitavaju)")
    print("  2. python prepare_dataset.py          (ponovo pripremi train/val/test split)")
    print("  3. python train_model.py ili finetune  (retreniraj model)")


if __name__ == "__main__":
    main()