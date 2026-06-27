"""
AR Air Hockey - MediaPipe Neon Edition (PvP)
============================================
TWO PLAYER:
  Left half  -> P1 (red)
  Right half -> P2 (blue)
  Point index finger to activate paddle.
  Q = quit  |  R = restart
"""

import cv2
import mediapipe as mp
import numpy as np
import joblib
import threading
import math
import random
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================================================
# CONFIG
# =========================================================

DW, DH = 1280, 720
CX      = DW // 2

GOAL_Y1 = DH // 2 - 130
GOAL_Y2 = DH // 2 + 130

MALLET_R  = 40
PUCK_R    = 24
WIN_SCORE = 7

C_P1      = (80,  80,  255)
C_P2      = (255, 80,  80)
C_LINE    = (60,  255, 60)
C_PUCK    = (255, 255, 255)
C_PUCK_GL = (180, 140, 255)

# Camera zoom
CAM_ZOOM  = 1.6


# Pre-allocated bloom buffer to avoid re-allocating memory every frame
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
    # Ambil dist_1 s/d dist_20 (buang dist_0 yg selalu 0.0, sesuai prepare_dataset.py)
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
        self.x   = float(DW // 2)
        self.y   = float(DH // 2)
        angle    = math.radians(random.uniform(-40, 40))
        speed    = self.BASE_SPEED
        self.vx  = direction * speed * math.cos(angle)
        self.vy  = speed * math.sin(angle)
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
    cv2.rectangle(ov, (0,        GOAL_Y1), (22,       GOAL_Y2), C_P1, -1)
    cv2.rectangle(ov, (DW - 22,  GOAL_Y1), (DW,       GOAL_Y2), C_P2, -1)
    bloom = cv2.blur(ov, (31, 31))
    return np.clip(
        ov.astype(np.float32) * 0.55 + bloom.astype(np.float32) * 0.6,
        0, 255
    ).astype(np.uint8)



def draw_bloom(frame, puck, m1, m2):
    """Applies a neon bloom effect using a pre-allocated buffer for performance."""
    _bloom_src[:] = 0   # Reset buffer in-place to save memory allocation cost

    for i, (tx, ty) in enumerate(puck.trail):
        a = (i + 1) / max(len(puck.trail), 1)
        cv2.circle(_bloom_src, (tx, ty), max(1, int(PUCK_R * 0.6 * a)),
                   (int(100*a), int(80*a), int(255*a)), -1)
    cv2.circle(_bloom_src, (int(puck.x), int(puck.y)), PUCK_R, C_PUCK_GL, -1)
    for m in (m1, m2):
        cv2.circle(_bloom_src, (int(m.x), int(m.y)), MALLET_R, m.color, -1)
    cv2.addWeighted(frame, 1.0, cv2.blur(_bloom_src, (45, 45)), 0.9, 0, dst=frame)


def draw_objects(frame, puck, m1, m2, speed_ratio):
    r = int(255)
    g = int(255 * (1 - speed_ratio * 0.6))
    b = int(255 * (1 - speed_ratio * 0.8))
    cv2.circle(frame, (int(puck.x), int(puck.y)), PUCK_R, (b, g, r), -1)
    cv2.circle(frame, (int(puck.x - 7), int(puck.y - 7)), 7, (255,255,255), -1)
    for m in (m1, m2):
        cv2.circle(frame, (int(m.x), int(m.y)), MALLET_R, m.color, -1)
        cv2.circle(frame, (int(m.x), int(m.y)), MALLET_R, (255,255,255), 2)


def draw_goal_flash(display, flash_frames):
    if flash_frames <= 0:
        return
    alpha  = flash_frames / 20 * 0.4
    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (DW, DH), (0, 200, 255), -1)
    cv2.addWeighted(overlay, alpha, display, 1 - alpha, 0, display)


def draw_hud(display, score1, score2, active1, active2, winner, hits, fps):
    cv2.putText(display, f"P1  {score1}", (DW//4 - 70, 50),
                cv2.FONT_HERSHEY_DUPLEX, 1.2, C_P1, 2)
    cv2.putText(display, f"P2  {score2}", (3*DW//4 - 70, 50),
                cv2.FONT_HERSHEY_DUPLEX, 1.2, C_P2, 2)
    cv2.putText(display, f"FPS  {fps:.1f}", (DW//2 - 50, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)

    if hits > 0:
        cv2.putText(display, f"rally  {hits}", (DW//2 - 55, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 80), 2)

    p1_label = "▶ ACTIVE" if active1 else "no gesture"
    p2_label = "▶ ACTIVE" if active2 else "no gesture"
    cv2.putText(display, f"P1: {p1_label}", (10, DH - 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_P1 if active1 else (80,80,80), 2)
    cv2.putText(display, f"P2: {p2_label}", (DW - 220, DH - 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_P2 if active2 else (80,80,80), 2)

    cv2.putText(display, "Q=quit  R=restart", (DW//2 - 110, DH - 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150,150,150), 1)

    if winner is not None:
        text = f"PLAYER {winner} WINS!"
        sz   = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 2, 4)[0]
        tx   = (DW - sz[0]) // 2
        cv2.rectangle(display, (tx-30, DH//2-70), (tx+sz[0]+30, DH//2+50), (0,0,0), -1)
        cv2.rectangle(display, (tx-30, DH//2-70), (tx+sz[0]+30, DH//2+50), (255,255,255), 2)
        cv2.putText(display, text, (tx, DH//2),
                    cv2.FONT_HERSHEY_DUPLEX, 2, (80,255,120), 4)
        cv2.putText(display, "Press R to restart", (tx+60, DH//2+40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200,200,200), 2)


def save_fps_figure(samples, output_path, title):
    if not samples:
        return

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(1, len(samples) + 1)
    ax.plot(x, samples, color="#4C72B0", linewidth=2, marker="o", markersize=4)
    ax.set_title(title)
    ax.set_xlabel("Second")
    ax.set_ylabel("FPS")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    mean_fps = float(np.mean(samples))
    min_fps = float(np.min(samples))
    max_fps = float(np.max(samples))
    summary = f"mean={mean_fps:.2f}, min={min_fps:.2f}, max={max_fps:.2f}"
    ax.text(
        0.02, 0.95, summary,
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    print(f"Saved: {output_path}")


# =========================================================
# MAIN
# =========================================================

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  DW)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DH)

    cam = CameraThread(cap)

    mp_hands = mp.solutions.hands
    hands    = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )
    mp_draw = mp.solutions.drawing_utils

    model  = joblib.load("model/gesture_model.pkl")
    scaler = joblib.load("model/scaler.pkl")

    puck    = Puck()
    p1      = Mallet("left",  C_P1)
    p2      = Mallet("right", C_P2)
    score1  = score2 = 0
    winner  = None
    table   = build_table()
    shake   = ScreenShake()
    active1 = active2 = False
    goal_flash = 0

    # State to throttle prediction: save the last gesture for each hand to reduce CPU usage
    last_gesture = {}   # key: hand index, value: 0 or 1
    frame_count  = 0
    fps = 0.0
    fps_frames = 0
    fps_last_time = time.perf_counter()
    fps_samples = []

    print("AR Air Hockey (PvP) — point your index finger to control your paddle!")

    while True:
        frame = cam.read()
        if frame is None:
            continue

        frame_count += 1
        fps_frames += 1

        # Calculate crop margins based on original frame dimensions to avoid double-resizing
        h_cam, w_cam = frame.shape[:2]
        margin_x = (1.0 - 1.0 / CAM_ZOOM) / 2.0
        margin_y = (1.0 - 1.0 / CAM_ZOOM) / 2.0

        now = time.perf_counter()
        elapsed = now - fps_last_time
        if elapsed >= 1.0:
            fps = fps_frames / elapsed
            fps_samples.append(fps)
            fps_frames = 0
            fps_last_time = now

        # MediaPipe tetap jalan pada full frame untuk deteksi terbaik
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        active1 = active2 = False

        if results.multi_hand_landmarks and winner is None:
            for idx, hlm in enumerate(results.multi_hand_landmarks):
                mp_draw.draw_landmarks(frame, hlm, mp_hands.HAND_CONNECTIONS)

                # Throttle inference: only predict on odd frames to save CPU
                if frame_count % 2 == 1:
                    feats   = landmarks_to_distances(hlm)
                    gesture = model.predict(feats)[0]
                    last_gesture[idx] = gesture
                else:
                    gesture = last_gesture.get(idx, 0)  # Use result from previous frame

                active = bool(gesture == 1)

                # Remap index finger coordinates directly from raw camera dimensions
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

        goal_flash = max(0, goal_flash - 1)

        speed_ratio = min(math.hypot(puck.vx, puck.vy) / Puck.MAX_SPEED, 1.0)

        # Crop directly from the raw camera frame (only resize once to DWxDH)
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
        draw_hud(display, score1, score2, active1, active2, winner, puck.hits, fps)

        # Screen shake
        ox, oy = shake.offset()
        if ox != 0 or oy != 0:
            M = np.float32([[1, 0, ox], [0, 1, oy]])
            display = cv2.warpAffine(display, M, (DW, DH))

        cv2.imshow("AR Air Hockey – PvP", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            puck         = Puck()
            score1       = score2 = 0
            winner       = None
            last_gesture = {}

    hands.close()
    cam.stop()
    cap.release()
    save_fps_figure(fps_samples, "model/fps_desktop.png", "Desktop FPS Over Time")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
