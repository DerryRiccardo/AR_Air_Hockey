"""
AR Air Hockey – Flask MJPEG Stream
===================================
Run:  python app.py
Open: http://localhost:5000

The game loop runs in a background thread.
Frames are JPEG-encoded and streamed to the browser via multipart/x-mixed-replace.
P1 controls left half, P2 controls right half – point index finger to activate paddle.
"""

import threading
import time
import math
import random

import cv2
import mediapipe as mp
import numpy as np
import joblib
from flask import Flask, Response, render_template_string, jsonify

# =========================================================
# CONFIG  (mirrors air_hockey.py)
# =========================================================

DW, DH    = 1280, 720
CX        = DW // 2
GOAL_Y1   = DH // 2 - 130
GOAL_Y2   = DH // 2 + 130
MALLET_R  = 40
PUCK_R    = 24
WIN_SCORE = 7

C_P1      = (80,  80,  255)
C_P2      = (255, 80,  80)
C_LINE    = (60,  255, 60)
C_PUCK_GL = (180, 140, 255)
CAM_ZOOM  = 1.6

_bloom_src = np.zeros((DH, DW, 3), dtype=np.uint8)

# =========================================================
# FEATURE EXTRACTION
# =========================================================

def landmarks_to_distances(hand_lms):
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms.landmark])
    wrist  = coords[0]
    dists  = np.linalg.norm(coords - wrist, axis=1)
    max_d  = dists.max()
    if max_d > 1e-6:
        dists /= max_d
    return dists[1:].reshape(1, -1)

# =========================================================
# CAMERA THREAD
# =========================================================

class CameraThread:
    def __init__(self, cap):
        self.cap     = cap
        self.frame   = None
        self.lock    = threading.Lock()
        self.running = True
        threading.Thread(target=self._update, daemon=True).start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                with self.lock:
                    self.frame = frame

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.running = False

# =========================================================
# SCREEN SHAKE
# =========================================================

class ScreenShake:
    def __init__(self):
        self.frames    = 0
        self.intensity = 0

    def trigger(self, intensity=18):
        self.frames    = 14
        self.intensity = intensity

    def offset(self):
        if self.frames <= 0:
            return 0, 0
        self.frames -= 1
        decay = self.frames / 14
        ox = int(random.uniform(-1, 1) * self.intensity * decay)
        oy = int(random.uniform(-1, 1) * self.intensity * decay)
        return ox, oy

# =========================================================
# PUCK
# =========================================================

class Puck:
    BASE_SPEED = 18.0
    MAX_SPEED  = 64.0
    HIT_BOOST  = 1.07

    def __init__(self):
        self.trail = []
        self.hits  = 0
        self.reset()

    def reset(self, direction=1):
        self.x  = float(DW // 2)
        self.y  = float(DH // 2)
        angle   = math.radians(random.uniform(-40, 40))
        speed   = self.BASE_SPEED
        self.vx = direction * speed * math.cos(angle)
        self.vy = speed * math.sin(angle)
        self.hits = 0
        self.trail.clear()

    def _speed(self):
        return math.hypot(self.vx, self.vy)

    def _cap(self, limit=None):
        limit = limit or self.MAX_SPEED
        s = self._speed()
        if s > limit:
            self.vx = self.vx / s * limit
            self.vy = self.vy / s * limit

    def update(self):
        self.trail.append((int(self.x), int(self.y)))
        if len(self.trail) > 22:
            self.trail.pop(0)
        self.x += self.vx
        self.y += self.vy
        self.vx *= 0.997
        self.vy *= 0.997
        self._cap()
        if self.y - PUCK_R < 0:
            self.y  = PUCK_R
            self.vy = abs(self.vy) * 1.02
        if self.y + PUCK_R > DH:
            self.y  = DH - PUCK_R
            self.vy = -abs(self.vy) * 1.02
        if self.x - PUCK_R < 0 and not (GOAL_Y1 <= self.y <= GOAL_Y2):
            self.x  = PUCK_R
            self.vx = abs(self.vx) * 1.02
        if self.x + PUCK_R > DW and not (GOAL_Y1 <= self.y <= GOAL_Y2):
            self.x  = DW - PUCK_R
            self.vx = -abs(self.vx) * 1.02

    def check_goal(self):
        if self.x - PUCK_R <= 0  and GOAL_Y1 <= self.y <= GOAL_Y2:
            return 2
        if self.x + PUCK_R >= DW and GOAL_Y1 <= self.y <= GOAL_Y2:
            return 1
        return 0

    def collide(self, mallet):
        dx   = self.x - mallet.x
        dy   = self.y - mallet.y
        dist = math.hypot(dx, dy)
        mind = PUCK_R + MALLET_R
        if 0 < dist < mind:
            nx, ny  = dx / dist, dy / dist
            self.x += nx * (mind - dist + 1)
            self.y += ny * (mind - dist + 1)
            dot     = self.vx * nx + self.vy * ny
            self.vx = self.vx - 2 * dot * nx + mallet.vx * 1.6
            self.vy = self.vy - 2 * dot * ny + mallet.vy * 1.6
            self.hits += 1
            boost = min(self.HIT_BOOST ** min(self.hits, 8), 1.8)
            self.vx *= boost
            self.vy *= boost
            s = self._speed()
            if s < 10:
                self.vx = nx * 10
                self.vy = ny * 10
            self._cap()
            return True
        return False

# =========================================================
# MALLET
# =========================================================

class Mallet:
    def __init__(self, side, color):
        self.side  = side
        self.color = color
        self.x     = float(DW // 4 if side == "left" else 3 * DW // 4)
        self.y     = float(DH // 2)
        self.vx    = 0.0
        self.vy    = 0.0

    def update(self, tx, ty):
        if self.side == "left":
            tx = np.clip(tx, MALLET_R, CX - MALLET_R)
        else:
            tx = np.clip(tx, CX + MALLET_R, DW - MALLET_R)
        ty      = np.clip(ty, MALLET_R, DH - MALLET_R)
        lerp    = 0.38
        nx      = self.x + (tx - self.x) * lerp
        ny      = self.y + (ty - self.y) * lerp
        self.vx = nx - self.x
        self.vy = ny - self.y
        self.x  = nx
        self.y  = ny

# =========================================================
# RENDERING
# =========================================================

def build_table():
    ov = np.zeros((DH, DW, 3), dtype=np.uint8)
    for y in range(0, DH, 28):
        cv2.line(ov, (CX, y), (CX, min(y + 14, DH)), C_LINE, 2)
    cv2.circle(ov, (DW // 2, DH // 2), 120, C_LINE, 2)
    cv2.rectangle(ov, (0,       GOAL_Y1), (22,      GOAL_Y2), C_P1, -1)
    cv2.rectangle(ov, (DW - 22, GOAL_Y1), (DW,      GOAL_Y2), C_P2, -1)
    bloom = cv2.blur(ov, (31, 31))
    return np.clip(
        ov.astype(np.float32) * 0.55 + bloom.astype(np.float32) * 0.6,
        0, 255
    ).astype(np.uint8)


def draw_bloom(frame, puck, m1, m2):
    _bloom_src[:] = 0
    for i, (tx, ty) in enumerate(puck.trail):
        a = (i + 1) / max(len(puck.trail), 1)
        cv2.circle(_bloom_src, (tx, ty), max(1, int(PUCK_R * 0.6 * a)),
                   (int(100*a), int(80*a), int(255*a)), -1)
    cv2.circle(_bloom_src, (int(puck.x), int(puck.y)), PUCK_R, C_PUCK_GL, -1)
    for m in (m1, m2):
        cv2.circle(_bloom_src, (int(m.x), int(m.y)), MALLET_R, m.color, -1)
    cv2.addWeighted(frame, 1.0, cv2.blur(_bloom_src, (45, 45)), 0.9, 0, dst=frame)


def draw_objects(frame, puck, m1, m2, speed_ratio):
    r = 255
    g = int(255 * (1 - speed_ratio * 0.6))
    b = int(255 * (1 - speed_ratio * 0.8))
    cv2.circle(frame, (int(puck.x), int(puck.y)), PUCK_R, (b, g, r), -1)
    cv2.circle(frame, (int(puck.x - 7), int(puck.y - 7)), 7, (255, 255, 255), -1)
    for m in (m1, m2):
        cv2.circle(frame, (int(m.x), int(m.y)), MALLET_R, m.color, -1)
        cv2.circle(frame, (int(m.x), int(m.y)), MALLET_R, (255, 255, 255), 2)


def draw_goal_flash(display, flash_frames):
    if flash_frames <= 0:
        return
    alpha   = flash_frames / 20 * 0.4
    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (DW, DH), (0, 200, 255), -1)
    cv2.addWeighted(overlay, alpha, display, 1 - alpha, 0, display)


def draw_hud(display, score1, score2, active1, active2, winner, hits):
    cv2.putText(display, f"P1  {score1}", (DW//4 - 70, 50),
                cv2.FONT_HERSHEY_DUPLEX, 1.2, C_P1, 2)
    cv2.putText(display, f"P2  {score2}", (3*DW//4 - 70, 50),
                cv2.FONT_HERSHEY_DUPLEX, 1.2, C_P2, 2)
    if hits > 0:
        cv2.putText(display, f"rally  {hits}", (DW//2 - 55, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 80), 2)
    p1_label = "ACTIVE" if active1 else "no gesture"
    p2_label = "ACTIVE" if active2 else "no gesture"
    cv2.putText(display, f"P1: {p1_label}", (10, DH - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_P1 if active1 else (80, 80, 80), 2)
    cv2.putText(display, f"P2: {p2_label}", (DW - 220, DH - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_P2 if active2 else (80, 80, 80), 2)
    if winner is not None:
        text = f"PLAYER {winner} WINS!"
        sz   = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 2, 4)[0]
        tx   = (DW - sz[0]) // 2
        cv2.rectangle(display, (tx-30, DH//2-70), (tx+sz[0]+30, DH//2+50), (0, 0, 0), -1)
        cv2.rectangle(display, (tx-30, DH//2-70), (tx+sz[0]+30, DH//2+50), (255, 255, 255), 2)
        cv2.putText(display, text, (tx, DH//2),
                    cv2.FONT_HERSHEY_DUPLEX, 2, (80, 255, 120), 4)
        cv2.putText(display, "Press Restart below", (tx+60, DH//2+40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)

# =========================================================
# GAME STATE  (shared between game thread + Flask routes)
# =========================================================

class GameState:
    def __init__(self):
        self.lock          = threading.Lock()
        self.running       = False
        self.restart_flag  = False
        self.latest_jpeg   = None   # bytes of the latest JPEG frame

state = GameState()


def game_loop():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  DW)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DH)

    cam = CameraThread(cap)

    mp_hands_mod = mp.solutions.hands
    hands = mp_hands_mod.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )
    mp_draw = mp.solutions.drawing_utils

    model  = joblib.load("model/gesture_model.pkl")
    scaler = joblib.load("model/scaler.pkl")

    puck         = Puck()
    p1           = Mallet("left",  C_P1)
    p2           = Mallet("right", C_P2)
    score1       = score2 = 0
    winner       = None
    table        = build_table()
    shake        = ScreenShake()
    active1      = active2 = False
    goal_flash   = 0
    last_gesture = {}
    frame_count  = 0

    margin_x = (1.0 - 1.0 / CAM_ZOOM) / 2.0
    margin_y = (1.0 - 1.0 / CAM_ZOOM) / 2.0

    while True:
        with state.lock:
            if not state.running:
                break

        frame = cam.read()
        if frame is None:
            time.sleep(0.005)
            continue

        frame_count += 1
        h_cam, w_cam = frame.shape[:2]

        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        active1 = active2 = False

        if results.multi_hand_landmarks and winner is None:
            for idx, hlm in enumerate(results.multi_hand_landmarks):
                mp_draw.draw_landmarks(frame, hlm, mp_hands_mod.HAND_CONNECTIONS)

                if frame_count % 2 == 1:
                    feats   = landmarks_to_distances(hlm)
                    gesture = model.predict(feats)[0]
                    last_gesture[idx] = gesture
                else:
                    gesture = last_gesture.get(idx, 0)

                active = bool(gesture == 1)
                raw_x  = hlm.landmark[8].x
                raw_y  = hlm.landmark[8].y
                tip_x  = int((raw_x - margin_x) * CAM_ZOOM * DW)
                tip_y  = int((raw_y - margin_y) * CAM_ZOOM * DH)

                if tip_x < CX:
                    active1 = active
                    if active:
                        p1.update(tip_x, tip_y)
                else:
                    active2 = active
                    if active:
                        p2.update(tip_x, tip_y)

        # Handle restart signal
        with state.lock:
            if state.restart_flag:
                puck         = Puck()
                score1       = score2 = 0
                winner       = None
                last_gesture = {}
                state.restart_flag = False

        # Physics
        if winner is None:
            puck.update()
            puck.collide(p1)
            puck.collide(p2)

            scored = puck.check_goal()
            if scored == 1:
                score1 += 1
                puck.reset(-1)
                shake.trigger(22)
                goal_flash = 20
            elif scored == 2:
                score2 += 1
                puck.reset(1)
                shake.trigger(22)
                goal_flash = 20

            if score1 >= WIN_SCORE:
                winner = 1
            elif score2 >= WIN_SCORE:
                winner = 2

        goal_flash  = max(0, goal_flash - 1)
        speed_ratio = min(math.hypot(puck.vx, puck.vy) / Puck.MAX_SPEED, 1.0)

        # Build display frame
        crop_x1 = int(margin_x * w_cam)
        crop_y1 = int(margin_y * h_cam)
        crop_x2 = w_cam - crop_x1
        crop_y2 = h_cam - crop_y1
        crop    = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        display = cv2.resize(crop, (DW, DH))
        display = (display * 0.55).astype(np.uint8)
        cv2.addWeighted(display, 1.0, table, 1.0, 0, display)

        draw_goal_flash(display, goal_flash)
        draw_bloom(display, puck, p1, p2)
        draw_objects(display, puck, p1, p2, speed_ratio)
        draw_hud(display, score1, score2, active1, active2, winner, puck.hits)

        ox, oy = shake.offset()
        if ox != 0 or oy != 0:
            M       = np.float32([[1, 0, ox], [0, 1, oy]])
            display = cv2.warpAffine(display, M, (DW, DH))

        # JPEG-encode and store for streaming
        ret, buf = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ret:
            with state.lock:
                state.latest_jpeg = buf.tobytes()

        time.sleep(0.001)

    hands.close()
    cam.stop()
    cap.release()
    with state.lock:
        state.latest_jpeg = None
        state.running     = False


# =========================================================
# FLASK
# =========================================================

app = Flask(__name__)

def _make_placeholder():
    img = np.zeros((DH, DW, 3), dtype=np.uint8)
    msg = "Game not running  –  press Start"
    sz  = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2)[0]
    cv2.putText(img, msg, ((DW - sz[0]) // 2, DH // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (70, 70, 90), 2)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()

PLACEHOLDER_JPEG = _make_placeholder()


def generate_frames():
    while True:
        with state.lock:
            jpeg = state.latest_jpeg or PLACEHOLDER_JPEG
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
        time.sleep(1 / 30)


HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AR Air Hockey</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    :root {
      --bg:      #080810;
      --surface: #0e0e1c;
      --border:  #1e1e38;
      --p1:      #5050ff;
      --p2:      #ff5050;
      --text:    #c8c8e8;
      --muted:   #44445a;
      --green:   #40ff80;
    }
    body {
      background: var(--bg);
      color: var(--text);
      font-family: 'Segoe UI', system-ui, sans-serif;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 20px 12px 32px;
      gap: 16px;
    }
    header { display: flex; align-items: center; gap: 14px; }
    header h1 {
      font-size: 1.4rem; font-weight: 700;
      letter-spacing: 2px; text-transform: uppercase;
      background: linear-gradient(90deg, var(--p1), #a060ff, var(--p2));
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .dot {
      width: 10px; height: 10px; border-radius: 50%;
      background: var(--muted); flex-shrink: 0; transition: background .4s;
    }
    .dot.live { background: var(--green); box-shadow: 0 0 8px var(--green); animation: blink 1.2s infinite; }
    @keyframes blink { 0%,100%{opacity:1} 50%{opacity:.3} }
    .stream-wrap {
      width: 100%; max-width: 960px;
      border-radius: 12px; overflow: hidden;
      border: 1px solid var(--border);
      box-shadow: 0 0 48px rgba(80,80,255,.12);
      background: #000; aspect-ratio: 16/9;
    }
    .stream-wrap img { width: 100%; height: 100%; object-fit: cover; display: block; }
    .controls { display: flex; gap: 12px; flex-wrap: wrap; justify-content: center; }
    button {
      padding: 11px 36px; border-radius: 50px; border: none;
      font-size: .9rem; font-weight: 600; cursor: pointer;
      letter-spacing: .5px; transition: transform .12s, opacity .12s;
    }
    button:disabled { opacity: .3; cursor: not-allowed; }
    button:not(:disabled):hover { transform: translateY(-2px); }
    #startBtn   { background: linear-gradient(135deg, var(--p1), #9050ff); color: #fff; box-shadow: 0 4px 20px rgba(80,80,255,.4); }
    #stopBtn    { background: linear-gradient(135deg, var(--p2), #ff8020); color: #fff; box-shadow: 0 4px 20px rgba(255,60,40,.35); }
    #restartBtn { background: var(--surface); color: var(--text); border: 1px solid var(--border); }
    .hint { font-size: .78rem; color: var(--muted); text-align: center; line-height: 1.9; }
    .hint span { color: #7070cc; }
    #toast {
      position: fixed; bottom: 24px; left: 50%;
      transform: translateX(-50%) translateY(16px);
      background: var(--surface); border: 1px solid var(--border);
      color: #a0a0ff; padding: 9px 22px; border-radius: 50px;
      font-size: .82rem; opacity: 0; transition: opacity .3s, transform .3s;
      pointer-events: none;
    }
    #toast.show { opacity: 1; transform: translateX(-50%) translateY(0); }
  </style>
</head>
<body>
  <header>
    <div class="dot" id="dot"></div>
    <h1>AR Air Hockey</h1>
  </header>

  <div class="stream-wrap">
    <img id="feed" src="/video_feed" alt="game stream">
  </div>

  <div class="controls">
    <button id="startBtn"   onclick="startGame()">&#9654; Start</button>
    <button id="stopBtn"    onclick="stopGame()"    disabled>&#9632; Stop</button>
    <button id="restartBtn" onclick="restartGame()" disabled>&#8635; Restart</button>
  </div>

  <p class="hint">
    Point your <span>index finger</span> at the camera to move your paddle.<br>
    <span>Left half of camera &rarr; P1 (blue)</span> &nbsp;&middot;&nbsp; <span>Right half &rarr; P2 (red)</span><br>
    First to <span>7 goals</span> wins &nbsp;&middot;&nbsp; Allow camera access when prompted by your browser.
  </p>

  <div id="toast"></div>

  <script>
    function toast(msg) {
      const t = document.getElementById('toast');
      t.textContent = msg;
      t.classList.add('show');
      setTimeout(() => t.classList.remove('show'), 2600);
    }
    function setRunning(running) {
      document.getElementById('dot').className        = 'dot' + (running ? ' live' : '');
      document.getElementById('startBtn').disabled   = running;
      document.getElementById('stopBtn').disabled    = !running;
      document.getElementById('restartBtn').disabled = !running;
    }
    async function startGame() {
      const d = await fetch('/start', { method: 'POST' }).then(r => r.json());
      if (d.ok) { setRunning(true); toast('Game started — show your hands!'); }
      else toast('Error: ' + d.error);
    }
    async function stopGame() {
      const d = await fetch('/stop', { method: 'POST' }).then(r => r.json());
      if (d.ok) { setRunning(false); toast('Game stopped.'); }
      else toast('Error: ' + d.error);
    }
    async function restartGame() {
      const d = await fetch('/restart', { method: 'POST' }).then(r => r.json());
      if (d.ok) toast('Game restarted!');
    }
    setInterval(async () => {
      const d = await fetch('/status').then(r => r.json());
      setRunning(d.running);
    }, 2000);
  </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/start", methods=["POST"])
def start():
    with state.lock:
        if state.running:
            return jsonify(ok=False, error="Already running.")
        state.running = True
    threading.Thread(target=game_loop, daemon=True).start()
    return jsonify(ok=True)


@app.route("/stop", methods=["POST"])
def stop():
    with state.lock:
        if not state.running:
            return jsonify(ok=False, error="Not running.")
        state.running = False
    return jsonify(ok=True)


@app.route("/restart", methods=["POST"])
def restart():
    with state.lock:
        if not state.running:
            return jsonify(ok=False, error="Game not running.")
        state.restart_flag = True
    return jsonify(ok=True)


@app.route("/status")
def status():
    with state.lock:
        running = state.running
    return jsonify(running=running)


if __name__ == "__main__":
    print("Open http://localhost:5000 in your browser")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)