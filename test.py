"""
Environment Smoke Test
======================
Runs a series of checks to ensure all runtime dependencies are met:
  1. MediaPipe installation and versions
  2. Model and scaler loading
  3. Inference shape compatibility
  4. End-to-end inference execution
  5. Camera access (index 0)

Usage:
  python test.py

Exit code 0 indicates success. Exit code 1 indicates failure.
"""

import sys

PASS = "[OK]"
FAIL = "[FAIL]"
WARN = "[WARN]"

all_ok = True

def check(label, fn):
    global all_ok
    try:
        result = fn()
        msg = f"  {result}" if result else ""
        print(f"{PASS} {label}{msg}")
        return True
    except Exception as e:
        print(f"{FAIL} {label}")
        print(f"       {e}")
        all_ok = False
        return False

# ── 1. MediaPipe ───────────────────────────────────────────────────────────────
print("\n── Dependencies ────────────────────────────────────────────────")

def check_mediapipe():
    import mediapipe as mp
    assert hasattr(mp, "solutions"), "mp.solutions not found"
    _ = mp.solutions.hands   # ensure submodule is accessible
    return f"v{mp.__version__}"

check("MediaPipe", check_mediapipe)

# ── 2. OpenCV & Camera ────────────────────────────────────────────────────────
def check_opencv():
    import cv2
    return f"cv2 v{cv2.__version__}"

check("OpenCV", check_opencv)

def check_camera():
    import cv2
    cap = cv2.VideoCapture(0)
    opened = cap.isOpened()
    if opened:
        ret, frame = cap.read()
        cap.release()
        assert ret and frame is not None, "Camera opened but frame is empty"
        h, w = frame.shape[:2]
        return f"camera index 0 OK ({w}x{h})"
    else:
        cap.release()
        raise RuntimeError("Cannot open camera at index 0")

check("Camera", check_camera)

# ── 3. NumPy & scikit-learn ───────────────────────────────────────────────────
def check_numpy():
    import numpy as np
    return f"v{np.__version__}"

def check_sklearn():
    import sklearn
    return f"v{sklearn.__version__}"

check("NumPy",      check_numpy)
check("scikit-learn", check_sklearn)

# ── 4. Model & scaler ─────────────────────────────────────────────────────────
print("\n── Model ───────────────────────────────────────────────────────")

def check_model_load():
    import joblib, os
    assert os.path.exists("model/gesture_model.pkl"), \
        "model/gesture_model.pkl not found — run train_model.py first"
    assert os.path.exists("model/scaler.pkl"), \
        "model/scaler.pkl not found — run train_model.py first"
    model  = joblib.load("model/gesture_model.pkl")
    scaler = joblib.load("model/scaler.pkl")
    return f"model={type(model).__name__}"

model_ok = check("Load model & scaler", check_model_load)

# ── 5. Inference shape ────────────────────────────────────────────────────────
def check_inference():
    import joblib
    import numpy as np

    model  = joblib.load("model/gesture_model.pkl")
    scaler = joblib.load("model/scaler.pkl")

    # Automatically detect the number of features expected by the model
    n_features = scaler.n_features_in_
    dummy      = np.zeros((1, n_features))
    pred       = model.predict(dummy)

    assert pred[0] in (0, 1), f"Invalid prediction output: {pred[0]}"
    return f"input shape ({1}, {n_features}) → class {pred[0]}"

if model_ok:
    check("Inference shape & output", check_inference)

# ── 6. MediaPipe hands end-to-end ────────────────────────────────────────────
print("\n── End-to-end ──────────────────────────────────────────────────")

def check_mediapipe_hands():
    import mediapipe as mp
    import numpy as np

    mp_hands = mp.solutions.hands
    hands    = mp_hands.Hands(max_num_hands=2,
                               min_detection_confidence=0.5,
                               min_tracking_confidence=0.5)

    # Small blank frame — no hands will be detected, but pipeline should run
    blank_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    result    = hands.process(blank_rgb)
    hands.close()

    detected = len(result.multi_hand_landmarks or [])
    return f"pipeline OK (hands detected on blank frame: {detected})"

check("MediaPipe Hands pipeline", check_mediapipe_hands)

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n────────────────────────────────────────────────────────────────")
if all_ok:
    print("All checks passed. Run: python air_hockey.py\n")
    sys.exit(0)
else:
    print("Some checks failed. Fix the errors above before running the game.\n")
    sys.exit(1)