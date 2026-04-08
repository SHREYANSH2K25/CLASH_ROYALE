import numpy as np
import time
import os
import cv2
import pyautogui
import threading
import torch
from ultralytics import YOLO
# from dotenv import load_dotenv
from actions import Actions



# EACH RECTANGLE PIXEL VALUE IS AROUND 21x26
# load_dotenv()

MAX_ENEMIES = 10
MAX_ALLIES = 10

INFER_W, INFER_H = 640, 640
CONF_THRES = 0.1
EMA_ALPHA = 0.4


SPELL_CARDS = {"Knight", "Archers", "Minions", "Goblins", "Spear Goblins", "Musketeer", "Giant","Mini P.E.K.K.A"}

TOWER_CLASSES = {
    "ally king tower",
    "ally princess tower",
    "enemy king tower",
    "enemy princess tower",
}



class ClashRoyaleEnv:
    def __init__(self):
        self.actions = Actions()

        # ── Local YOLO models (replaces Roboflow HTTP client) ──────────────────
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[ClashRoyaleEnv] Using device: {self.device}")
        base = os.path.dirname(__file__)
        self.model = YOLO(os.path.join(base, "YOLO_MODEL", "troop6.pt"))
        self.card_model = YOLO("YOLO_MODEL/card.pt").to(self.device)
        # ── Per-step frame / detection cache ───────────────────────────────────
        self._cached_frame   = None   # raw numpy frame
        self._cached_results = None   # list of (x1,y1,x2,y2,conf,cls) tuples
        # from inference_sdk import InferenceHTTPClient

        # self.rf_client = InferenceHTTPClient(
        #     api_url="https://detect.roboflow.com",
        #     api_key="YOUR_API_KEY"
        # )
        # ── EMA state smoother ─────────────────────────────────────────────────
        self._ema_state = None

        # ── Action space ───────────────────────────────────────────────────────
        self.num_cards   = 4
        self.grid_width  = 18
        self.grid_height = 28
        self.state_size  = 1 + 2 * (MAX_ALLIES + MAX_ENEMIES)
        
        self.available_actions = self._build_action_space()
        self.action_size       = len(self.available_actions)
        self.current_cards     = []

        # ── End-game watcher thread ────────────────────────────────────────────
        self.game_over_flag          = None
        self._endgame_thread         = None
        self._endgame_thread_stop    = threading.Event()

        # ── Reward state ───────────────────────────────────────────────────────
        self.prev_elixir                = None
        self.prev_enemy_presence        = None
        self.prev_enemy_princess_towers = None
        self.match_over_detected        = False

    # ══════════════════════════════════════════════════════════════════════════
    # FRAME CAPTURE  (in-memory, no disk I/O)
    # ══════════════════════════════════════════════════════════════════════════

    def _capture_frame(self):
        """Return cached numpy frame for this step, capturing once if needed."""
        if self._cached_frame is not None:
            return self._cached_frame
        # Actions.capture_area() must return a numpy BGR frame (not write to disk)
        frame = self.actions.capture_area()
        self._cached_frame = frame
        return frame

    # ══════════════════════════════════════════════════════════════════════════
    # YOLO DETECTION  (single inference per step via cache)
    # ══════════════════════════════════════════════════════════════════════════
    # def setup_roboflow(self):
    #     api_key = os.getenv('ROBOFLOW_API_KEY')
    #     if not api_key:
    #         raise ValueError("ROBOFLOW_API_KEY environment variable is not set. Please check your .env file.")
        
    #     return InferenceHTTPClient(
    #         api_url="http://localhost:9001",
    #         api_key=api_key
    #     )

    # return number of enemy and number of ally and,
    def fxn(self,a,b):
        var_a = 4
        var_b = 0.03
        var_c = 0.2
        # if 775-a > 
        return (1/(1+ np.exp(var_b*(456-a)))) > var_c

    def _run_detection(self):
        """Run YOLO on the current frame once per step, return detection list."""
        # detect = []
        # detect.append((
        #     1500,400, 1650, 550,
        #     0.9,
        #     "enemy Knight",
        # ))
        # return detect
        if self._cached_results is not None:
            return self._cached_results

        frame = self._capture_frame()
        frame = self._capture_frame()
        # frame = cv2.imread(img)
        frame = np.ascontiguousarray(frame)   # 🔥 FIX
        h, w  = frame.shape[:2]
        # cv2.imshow("frame", frame)
        # cv2.waitKey(0)
        resized = cv2.resize(frame, (INFER_W, INFER_H))
        results = self.model(resized, device=self.device, verbose=False)[0]
        # print("results",results)
        scale_x = w / INFER_W
        scale_y = h / INFER_H

        detections = []
        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < CONF_THRES:
                continue

            cls = results.names[int(box.cls[0])].lower().strip()
            x1, y1, x2, y2 = box.xyxy[0]

            # 🔥 SCALE FIRST
            x1 = int(float(x1) * scale_x)
            y1 = int(float(y1) * scale_y)
            x2 = int(float(x2) * scale_x)
            y2 = int(float(y2) * scale_y)

            detections.append((x1, y1, x2, y2, conf, cls))

            color = (0, 255, 0) if "ally" in cls else (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"{cls} {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 🔥 SHOW RESULT
        # print("YOLO_DETECT")
        # cv2.imshow("YOLO Detection", frame)
        # cv2.waitKey()   # use 1 for real-time, 0 for pause
        # cv2.destroyAllWindows()
        self._cached_results = detections
        return detections

    def _get_state(self):
        elixir     = self.actions.count_elixir()
        print(elixir)
        detections = self._run_detection()
        # return 

        allies, enemies = [], []
        
        # position_x,y,x2,y2, confidence, and class_(if ally or enemy)

        for x1, y1, x2, y2, conf, cls in detections:
            if cls in TOWER_CLASSES:
                continue
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            if self.fxn(cx,cy):
                allies.append((cx, cy))
            else:
                enemies.append((cx, cy))
        
        def pad(units, n):
            units = units[:n]
            units += [(0.0, 0.0)] * (n - len(units))
            return units

        allies  = pad(allies,  MAX_ALLIES)
        enemies = pad(enemies, MAX_ENEMIES)

        raw = []
        for x, y in allies + enemies:
            raw.append(x / self.actions.WIDTH)
            raw.append(y / self.actions.HEIGHT)
        raw = np.array(raw, dtype=np.float32)

        # exponential moving averages smoothing
        # instead of taking present states as final values blend it with previous one for a smooth transition
        if self._ema_state is None:
            self._ema_state = raw
            # 40% past and 60% present
        else:
            self._ema_state = EMA_ALPHA * self._ema_state + (1 - EMA_ALPHA) * raw
# elixir in 0-1 for better nn conversion 
        # return allies , enemies
        return np.concatenate(([elixir / 10.0], self._ema_state))



    # ══════════════════════════════════════════════════════════════════════════ 
    # RESET
    # ══════════════════════════════════════════════════════════════════════════ 

    # def _buffer(self):



    def reset(self):
        time.sleep(3)

        self.game_over_flag       = None
        self.match_over_detected  = False
        self._cached_frame        = None
        self._cached_results      = None
        self._ema_state           = None
        self.prev_elixir          = None
        self.prev_enemy_presence  = None

        self._endgame_thread_stop.clear()
        self._endgame_thread = threading.Thread(
            target=self._endgame_watcher, daemon=True
        )
        self._endgame_thread.start()

        self.prev_enemy_princess_towers = self._count_enemy_princess_towers()
        # return
        return self._get_state()



    def close(self):
        self._endgame_thread_stop.set()
        if self._endgame_thread:
            self._endgame_thread.join()

    # ══════════════════════════════════════════════════════════════════════════
    # STEP
    # ══════════════════════════════════════════════════════════════════════════



    def step(self, action_index):
        # Invalidate per-step cache
        self._cached_frame   = None
        self._cached_results = None

        # ── Match-over guard ───────────────────────────────────────────────────
        if (
            not self.match_over_detected
            and hasattr(self.actions, "detect_match_over")
            and self.actions.detect_match_over()
        ):
            print("Match over detected — forcing no-op until next game.")
            self.match_over_detected = True

        if self.match_over_detected:
            action_index = len(self.available_actions) - 1  # no-op

        # ── Episode-end check ─────────────────────────────────────────────────
        if self.game_over_flag:
            state  = self._get_state()
            reward = self._compute_reward(state)
            result = self.game_over_flag

            if result == "victory":
                reward += 100
                print("Victory — ending episode.")
            elif result == "defeat":
                reward -= 100
                print("Defeat — ending episode.")

            self.match_over_detected = False
            return state, reward, True

        # ── Card detection ────────────────────────────────────────────────────
        self.current_cards = self.detect_cards_in_hand()
        print("\nCards in hand:", self.current_cards)

        if all(card == "Unknown" for card in self.current_cards):
            print("All cards Unknown — clicking fallback position.")
            pyautogui.moveTo(1611, 800, duration=0.2)
            pyautogui.click()
            return self._get_state(), 0, False

        # ── Execute action ────────────────────────────────────────────────────
        action = self.available_actions[action_index]
        card_index, x_frac, y_frac = action
        print(f"Action: card={card_index}  x={x_frac:.2f}  y={y_frac:.2f}")

        spell_penalty = 0

        if card_index != -1 and card_index < len(self.current_cards):
            card_name = self.current_cards[card_index]
            print(f"Playing: {card_name}")
            x = int(x_frac * 472)  + 1368
            y = int(y_frac * 637) + 137
            print(x,y,card_index)
            self.actions.card_play(x, y, card_index)
            time.sleep(1)

            # Spell-on-empty-space penalty
            if card_name in SPELL_CARDS:
                state          = self._get_state()
                enemy_offset   = 1 + 2 * MAX_ALLIES
                enemy_flat     = state[enemy_offset:]
                enemy_positions = [
                    (
                        int(enemy_flat[i]     * self.actions.WIDTH),
                        int(enemy_flat[i + 1] * self.actions.HEIGHT),
                    )
                    for i in range(0, len(enemy_flat), 2)
                    if enemy_flat[i] != 0.0 or enemy_flat[i + 1] != 0.0
                ]
                radius      = 100
                hit_enemy   = any(
                    ((ex - x) ** 2 + (ey - y) ** 2) ** 0.5 < radius
                    for ex, ey in enemy_positions
                )
                if not hit_enemy:
                    spell_penalty = -5

        # ── Princess-tower reward ─────────────────────────────────────────────
        cur_towers         = self._count_enemy_princess_towers()
        tower_reward       = 0
        if self.prev_enemy_princess_towers is not None:
            if cur_towers < self.prev_enemy_princess_towers:
                tower_reward = 20
        self.prev_enemy_princess_towers = cur_towers

        # ── Compute reward / next state ───────────────────────────────────────
        next_state = self._get_state()
        reward     = self._compute_reward(next_state) + spell_penalty + tower_reward

        return next_state, reward, False



# DEPENDS CHANGE DYNAMICALLY
    def _compute_reward(self, state):
        if state is None:
            return 0.0

        elixir          = state[0] * 10
        enemy_flat      = state[1 + 2 * MAX_ALLIES:]
        enemy_presence  = float(np.sum(enemy_flat[1::2]))  # y-coords only
        reward          = -enemy_presence

        if self.prev_elixir is not None and self.prev_enemy_presence is not None:
            spent   = self.prev_elixir - elixir
            reduced = self.prev_enemy_presence - enemy_presence
            if spent > 0 and reduced > 0:
                reward += 2.0 * min(spent, reduced)

        self.prev_elixir         = elixir
        self.prev_enemy_presence = enemy_presence

        return reward


#  DONE
# just add that if card grey not playable 
    def detect_cards_in_hand(self):
        self.actions.capture_individual_cards()
        try:
            cards = []

            for i in range(1, 5):
                img_path = os.path.join(os.path.dirname(__file__), "screenshots", f"card_{i}.png")

                img = cv2.imread(img_path)
                if img is None:
                    print(f"Failed to load: {img_path}")
                    cards.append("Unknown")
                    continue
                # # --- GRAYSCALE CHECK ---
                # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                # saturation = hsv[:, :, 1]
                # avg_sat = saturation.mean()

                # # RGB channel difference check
                # b, g, r = cv2.split(img)
                # color_diff = (abs(r - g) + abs(g - b) + abs(b - r)).mean()

                # # FINAL DECISION
                # if avg_sat < 25 and color_diff < 20:
                #     cards.append("Unknown")
                #     continue
                res = self.card_model(img, device=self.device, verbose=False)[0]

                # classification output
                top1 = res.probs.top1
                confidence = float(res.probs.top1conf)

                if confidence < 0.4:
                    cards.append("Unknown")
                else:
                    cards.append(self.card_model.names[top1])

            print("Detected cards:", cards)
            return cards

        except Exception as exc:
            print(f"detect_cards_in_hand error: {exc}")
            return []
    # ══════════════════════════════════════════════════════════════════════════
    # PRINCESS TOWER COUNT  (uses cached detection when available)
    # ══════════════════════════════════════════════════════════════════════════


# DONE
    def _count_enemy_princess_towers(self):
        frame1 =  pyautogui.screenshot(region=(
            # self.actions.TOP_LEFT_X + int(0.05*self.actions.WIDTH), 
            # self.actions.TOP_LEFT_Y + int(0.16*self.actions.HEIGHT), 
            # int(0.25*self.actions.WIDTH), 
            # int(0.15*self.actions.HEIGHT)
            1380,
            196,
            140,
            150
        ))
        frame1 = np.array(frame1)
        frame2 =  pyautogui.screenshot(region=(
            1680,
            196,
            140,
            150
        ))
        frame2 = np.array(frame2)
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
        alive_template = cv2.imread("check_images/Alive_to.png",0)
        # destroyed_template = cv2.imread("check_images/dead_to.png",0)
        # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # gray_alive = cv2.cvtColor(alive_template, cv2.COLOR_BGR2RGB)
        # destroyed_template = cv2.cvtColor(destroyed_template, cv2.COLOR_BGR2RGB)
        # cv2.imshow("frame2",frame2)
        # cv2.imshow("frame1",frame1)
        # cv2.imshow("al",alive_template)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        result_alive1 = cv2.matchTemplate(frame1, alive_template, cv2.TM_CCOEFF_NORMED)
        result_alive2 = cv2.matchTemplate(frame2, alive_template, cv2.TM_CCOEFF_NORMED)
        # threshold = 0.7
        ans = 0
        # print(np.max(result_alive1))
        # print(np.max(result_alive2))
        if(np.max(result_alive1)> 0.6):
            ans+=1
        if(np.max(result_alive2)> 0.6):
            ans+=1
        # locations = np.where(result_alive >= threshold)
        # print(locations)
        # alive_count = len(list(zip(*locations[::-1])))
        # print(alive_count)
        return ans



    # ══════════════════════════════════════════════════════════════════════════
    # END-GAME WATCHER THREAD
    # ══════════════════════════════════════════════════════════════════════════


# DONE
    def _endgame_watcher(self):
        while not self._endgame_thread_stop.is_set():
            result = self.actions.detect_game_end()
            if result:
                self.game_over_flag = result
                break
            time.sleep(0.5)

    # ══════════════════════════════════════════════════════════════════════════
    # ACTION SPACE
    # ══════════════════════════════════════════════════════════════════════════

# DONE
    def _build_action_space(self):
        actions = [
            [card, x / (self.grid_width - 1), y / (self.grid_height - 1)]
            for card in range(self.num_cards)
            for x in range(self.grid_width)
            for y in range(self.grid_height)
        ]
        actions.append([-1, 0, 0])  # no-op
        return actions

    # keep old name as alias for any existing callers

#DONE
    def get_available_actions(self):
        return self.available_actions


if __name__ ==  '__main__':
    env = ClashRoyaleEnv()
    # print(env._count_enemy_princess_towers())
    # print(env.detect_cards_in_hand())
    # a,b,c = env.step(1)
    # print(a)
    # print(b)
    # print(c)
    # env._run_detection(r"C:\Users\shour\Downloads\WhatsApp Image 2026-04-05 at 11.59.23 PM.jpeg")
    print(env.fxn(120,0))
    # print(env._build_action_space())
    # val = pyautogui.center()
    # screenshot = pyautogui.screenshot(region=(
    #         1368,
    #         # 137,
    #         475,
    #         472,
    #         # 637,
    #         300
    # ))
    # screenshot.save(r"C:\Users\shour\RL\CLASH_ROYALE\images\scr3.jpg")
