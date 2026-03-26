import numpy as np
import time
import os
import cv2
import pyautogui
import threading
import torch
from ultralytics import YOLO
from Actions import Actions

MAX_ENEMIES = 10
MAX_ALLIES = 10

INFER_W, INFER_H = 640, 640
CONF_THRES = 0.35
EMA_ALPHA = 0.4

TOWER_CLASSES = {
    "ally king tower",
    "ally princess tower",
    "enemy king tower",
    "enemy princess tower"
}

SPELL_CARDS = {
    "Fireball", "Zap", "Arrows", "Tornado",
    "Rocket", "Lightning", "Freeze"
}


class ClashRoyaleEnv:
    def __init__(self):
        self.actions = Actions()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Device:", self.device)

        self.model = YOLO("models/troop.pt").to(self.device)
        self.card_model = YOLO("models/card.pt").to(self.device)

        self._cached_frame = None
        self._cached_results = None
        self._ema_state = None

        self.prev_elixir = None
        self.prev_enemy_presence = None

    # =========================
    # FRAME CAPTURE (FIXED)
    # =========================
    def _capture_frame(self):
        if self._cached_frame is not None:
            return self._cached_frame

        frame = self.actions.capture_area()  # MUST return frame now
        self._cached_frame = frame
        return frame

    # =========================
    # YOLO (SINGLE INFERENCE)
    # =========================
    def _run_detection(self):
        if self._cached_results is not None:
            return self._cached_results

        frame = self._capture_frame()
        h, w = frame.shape[:2]

        resized = cv2.resize(frame, (INFER_W, INFER_H))
        results = self.model(resized, device=self.device, verbose=False)[0]

        scale_x = w / INFER_W
        scale_y = h / INFER_H

        detections = []

        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < CONF_THRES:
                continue

            cls = results.names[int(box.cls[0])].lower().strip()

            x1, y1, x2, y2 = box.xyxy[0]
            x1 *= scale_x; x2 *= scale_x
            y1 *= scale_y; y2 *= scale_y

            detections.append((x1, y1, x2, y2, conf, cls))

        self._cached_results = detections
        return detections

    # =========================
    # STATE
    # =========================
    def _get_state(self):
        frame = self._capture_frame()
        elixir = self.actions.count_elixir()

        detections = self._run_detection()

        allies, enemies = [], []

        for x1, y1, x2, y2, conf, cls in detections:
            if cls in TOWER_CLASSES:
                continue

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            # safer handling
            if "ally" in cls:
                allies.append((cx, cy))
            elif "enemy" in cls:
                enemies.append((cx, cy))

        allies = allies[:MAX_ALLIES]
        enemies = enemies[:MAX_ENEMIES]

        def pad(units, n):
            units += [(0, 0)] * (n - len(units))
            return units[:n]

        allies = pad(allies, MAX_ALLIES)
        enemies = pad(enemies, MAX_ENEMIES)

        state = []

        for x, y in allies + enemies:
            state.append(x / self.actions.WIDTH)
            state.append(y / self.actions.HEIGHT)

        state = np.array(state, dtype=np.float32)

        # EMA FIX
        if self._ema_state is None:
            self._ema_state = state
        else:
            self._ema_state = EMA_ALPHA * self._ema_state + (1 - EMA_ALPHA) * state

        return np.concatenate(([elixir / 10.0], self._ema_state))

    # =========================
    # STEP
    # =========================
    def step(self, action):
        self._cached_frame = None
        self._cached_results = None

        state = self._get_state()
        reward = self._compute_reward(state)

        return state, reward, False

    # =========================
    # REWARD
    # =========================
    def _compute_reward(self, state):
        elixir = state[0] * 10
        enemy = state[1 + 2 * MAX_ALLIES:]

        enemy_presence = sum(enemy[1::2])
        reward = -enemy_presence

        if self.prev_elixir is not None:
            spent = self.prev_elixir - elixir
            reduced = self.prev_enemy_presence - enemy_presence

            if spent > 0 and reduced > 0:
                reward += 2 * min(spent, reduced)

        self.prev_elixir = elixir
        self.prev_enemy_presence = enemy_presence

        return reward

    # =========================
    # CARD DETECTION (FIXED)
    # =========================
    def detect_cards(self, images):
        cards = []

        for img in images:
            resized = cv2.resize(img, (INFER_W, INFER_H))
            res = self.card_model(resized, device=self.device, verbose=False)[0]

            if not res.boxes:
                cards.append("Unknown")
                continue

            best = max(res.boxes, key=lambda b: float(b.conf[0]))
            if float(best.conf[0]) < 0.4:
                cards.append("Unknown")
                continue

            cls = res.names[int(best.cls[0])]
            cards.append(cls)

        return cards