# AR Air Hockey — Computer Vision Project

Fingertip-tracked augmented-reality air hockey using MediaPipe hand detection
and a trained gesture classifier.

---

## Project structure

```
.
├── prepare_dataset.py # Step 1 – prepare gesture dataset (from Zenodo CSV)
├── train_model.py     # Step 2 – train & evaluate the MLP classifier
├── air_hockey.py      # Step 3 – run the game
├── requirements.txt
├── data/
│   ├── hand-gestures.csv  (input CSV; download link in prepare_dataset.py)
│   └── gesture_data.csv   (created by prepare_dataset.py)
└── model/
    ├── gesture_model.pkl  (created by train_model.py)
    ├── scaler.pkl
    ├── confusion_matrix.png
    └── training_curve.png
```

---

## How it works

### 1  Hand detection — MediaPipe Hands (pre-trained)
MediaPipe's `Hands` solution is itself a trained neural network (a lightweight
MobileNet-based pipeline) that returns 21 3-D landmarks for each hand in the
frame.  Landmark 8 is the index fingertip — that's the paddle position.

### 2  Gesture classification — MLP trained by you
A two-class MLP (128 → 64 → 2) is trained on the 63-dimensional landmark
vector (21 joints × x, y, z).

| Class | Gesture         | Game effect            |
|-------|-----------------|------------------------|
| 0     | Closed fist     | Paddle freezes         |
| 1     | Index finger up | Paddle tracks fingertip|

Training also fits a Random Forest baseline for comparison and prints a
5-fold cross-validation accuracy report.

### 3  Game
- Your paddle is constrained to the left half of the screen.
- A simple AI tracks the puck's y-position on the right half.
- Puck physics include wall bouncing, paddle-velocity transfer, and a speed cap.
- The webcam feed is used as the table background (AR effect).
- First to 7 goals wins.

---

## Setup (local — recommended, you have an RTX 2050)

```bash
# 1  Create a virtual environment
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# 2  Install dependencies
pip install -r requirements.txt

# 3  Prepare dataset
python prepare_dataset.py

# 4  Train the model
python train_model.py
# Outputs accuracy report + confusion matrix + training curve

# 5  Play
python air_hockey.py
```

---

## Running on Google Colab (alternative)

Colab does not support `cv2.imshow`.  Use this workaround:

```python
# At the top of any script, replace cv2.imshow with:
from google.colab.patches import cv2_imshow
```

Webcam capture in Colab requires a JavaScript snippet to grab a single frame.
For real-time play, run locally — Colab adds too much latency.

---

## University report notes

- **Model architecture**: 2-hidden-layer MLP (128, 64 neurons), ReLU activations, Adam optimiser, early stopping on validation loss.
- **Feature engineering**: Raw landmark coordinates are normalised with `StandardScaler` (zero mean, unit variance) before training.
- **Baseline**: Random Forest with 100 trees (no scaling needed).
- **Evaluation**: 80/20 stratified train-test split + 5-fold cross-validation.
- **Saved artefacts**: `model/confusion_matrix.png` and `model/training_curve.png` are ready to drop into your report.
