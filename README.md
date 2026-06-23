# Serbian Sign Language Recognition

A project for recognizing letters of the Serbian alphabet performed in sign language (SSL — Serbian Sign Language), based on hand/body motion captured from a webcam. The pipeline uses **MediaPipe Holistic** to extract body and hand landmarks from video, and an **LSTM neural network** to classify a 40-frame sequence as one of the alphabet letters. The project also includes tools for recording your own samples and for real-time recognition via webcam.

The dataset is sourced from [mlradak/serbian-sign-language](https://github.com/mlradak/serbian-sign-language), where each sign is stored as a `.npy` file containing a sequence of frames with body and hand keypoints (33 body points × 4 values + 21 points per hand × 3 values = 258 values per frame).

## How it works

The project follows a clear pipeline: **data collection/loading → preprocessing & feature extraction → model training → hyperparameter tuning → evaluation → real-time recognition**.

- **`record_samples.py`** — Captures hand/body landmarks from the webcam using MediaPipe Holistic. Uses a simple state machine (`waiting` → `recording`) that starts recording automatically once a hand enters the frame, captures exactly 40 frames, and saves them as a `.npy` file under `data/<LETTER>/`. Discards the recording if the hand leaves the frame too early.

- **`load_and_explore_data.py`** — Loads raw `.npy` recordings from `data/` and reshapes the flat arrays into per-frame sequences. Prints per-letter sample counts and frame-length statistics, and saves a summary plot (`dataset_analysis.png`).

- **`interpolation.py`** — Fixes frames where a hand wasn't detected (filled with zeros) by linearly interpolating between the last known and next known hand position.

- **`body_normalization.py`** — Normalizes coordinates relative to the shoulders (center + scale), so the model isn't affected by the signer's distance from the camera or position in frame.

- **`curl_smoothing.py`** — Applies a rolling average to smooth the finger-curl features over time, reducing frame-to-frame noise.

- **`prepare_dataset.py`** — Combines all preprocessing steps: interpolates gaps, normalizes coordinates, computes 10 additional **finger-curl features** (tip-to-wrist vs. base-to-wrist distance ratio for each finger on both hands), scales everything with `MinMaxScaler`, encodes labels, and splits the data into train/validation/test sets (stratified). Saves everything to `models/`.

- **`train_model.py`** — Trains an LSTM model (`LSTM(128) → Dropout → LSTM(64) → Dropout → Dense(64) → Dense(softmax)`) with early stopping and checkpointing on validation accuracy. Saves the best model to `models/best_model.keras` and a training curve plot (`training_results.png`).

- **`tune_model.py`** — Uses Keras Tuner's Hyperband algorithm to search over LSTM unit counts, dropout rates, dense layer size, and learning rate, then reports the best configuration.

- **`evaluate.py`** — Loads the trained model and test set, prints overall accuracy and a per-class classification report, flags letters with F1 < 0.85, lists the most common misclassifications, and saves a per-class F1 plot (`class_evaluation.png`).

- **`realtime.py`** — Real-time recognition demo. Captures 40 frames of a sign from the webcam, runs them through the same preprocessing pipeline used during training, and predicts the letter with a confidence score (shows top-3 predictions). Lets you build a word from recognized letters using keyboard shortcuts ( BACKSPACE to delete, SPACE to reset, ESC to quit).

- **`test.py`** — Small utility script to inspect the shape/contents of a single `.npy` file.

## Project structure

```
serbian-sign-language-recognition/
├── data/                       # raw .npy recordings, organized by letter folders
├── models/                     # generated train/val/test sets, scaler, trained models
├── record_samples.py
├── load_and_explore_data.py
├── body_normalization.py
├── interpolation.py
├── curl_smoothing.py
├── prepare_dataset.py
├── train_model.py
├── tune_model.py
├── evaluate.py
├── realtime.py
├── test.py
└── requirements.txt
```

## Setup & Usage

```bash
git clone https://github.com/kajadim/serbian-sign-language-recognition.git
cd serbian-sign-language-recognition

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

**1. Get data** — either download recordings from [mlradak/serbian-sign-language](https://github.com/mlradak/serbian-sign-language) into `data/<LETTER>/` folders, or record your own:

```bash
python record_samples.py A
```

**2. (Optional) Explore the dataset:**

```bash
python load_and_explore_data.py
```

**3. Prepare the training data:**

```bash
python prepare_dataset.py
```

**4. Train the model:**

```bash
python train_model.py
```

**5. (Optional) Tune hyperparameters:**

```bash
python tune_model.py
```

**6. Evaluate the model:**

```bash
python evaluate.py
```

**7. Run real-time recognition:**

```bash
python realtime.py
```

> Requires `models/best_model.keras` and `models/scaler.pkl` to exist (produced in steps 3–4) before running `realtime.py`.

**Requirements:** Python 3.10/3.11, a webcam, and the packages listed in `requirements.txt` (TensorFlow, Keras, MediaPipe, OpenCV, scikit-learn, etc.).
