import numpy as np
import time
import os
import cv2
import pyautogui
import threading
from dotenv import load_dotenv
from actions import Actions
import ssl
import urllib3
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from inference_sdk import InferenceHTTPClient
import pygetwindow as gw

# Load environment variables from .env file
load_dotenv()

MAX_ENEMIES = 10
MAX_ALLIES  = 10

SPELL_CARDS = ["Fireball", "Zap", "Arrows", "Tornado", "Rocket", "Lightning", "Freeze"]

# Roboflow hosted model for troop/tower detection
TROOP_MODEL_ID = "clash-royale-troop-detection/12"


class ClashRoyaleEnv:
    def __init__(self):
        # ── Detect BlueStacks window ──────────────────────────────────────────
        windows = gw.getWindowsWithTitle("BlueStacks")
        if not windows:
            raise RuntimeError("BlueStacks window not found. Make sure BlueStacks is open.")
        win = windows[0]
        self.actions = Actions(
            win.left,
            win.top,
            win.left + win.width,
            win.top + win.height
        )

        # ── Models ────────────────────────────────────────────────────────────
        self.rf_model = self._setup_roboflow()   # troop/tower detection (hosted)

        # ── State / action space ──────────────────────────────────────────────
        self.state_size   = 1 + 2 * (MAX_ALLIES + MAX_ENEMIES)
        self.num_cards    = 4
        self.grid_width   = 18
        self.grid_height  = 28

        # ── Screenshot paths ──────────────────────────────────────────────────
        self.script_dir      = os.path.dirname(__file__)
        self.screenshot_path = os.path.join(self.script_dir, 'screenshots', 'current.png')
        os.makedirs(os.path.dirname(self.screenshot_path), exist_ok=True)

        # ── Card reference templates (screenshots/card_1.png … card_4.png) ───
        # These are loaded once and used for template-matching during gameplay.
        self.card_templates = self._load_card_templates()

        self.available_actions = self.get_available_actions()
        self.action_size       = len(self.available_actions)
        self.current_cards     = []

        self.game_over_flag         = None
        self._endgame_thread        = None
        self._endgame_thread_stop   = threading.Event()

        self.prev_elixir             = None
        self.prev_enemy_presence     = None
        self.prev_enemy_princess_towers = None
        self.match_over_detected     = False

    # ── Setup helpers ─────────────────────────────────────────────────────────

    def _setup_roboflow(self):
        """Connect to the Roboflow serverless endpoint for troop detection."""
        api_key = os.getenv('ROBOFLOW_API_KEY')
        if not api_key:
            raise ValueError("ROBOFLOW_API_KEY is not set in your .env file.")
        return InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=api_key
        )

    def _load_card_templates(self):
        """
        Load reference card images from screenshots/card_1.png … card_4.png.
        Returns a dict: {slot_index (0-3): {"name": filename, "img": grayscale ndarray}}
        """
        templates = {}
        for i in range(1, 5):
            path = os.path.join(self.script_dir, 'screenshots', f'card_{i}.png')
            if os.path.exists(path):
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    templates[i - 1] = {"name": f"card_{i}", "img": img}
                    print(f"Loaded card template: card_{i}.png")
                else:
                    print(f"Warning: could not read {path}")
            else:
                print(f"Warning: template not found at {path}")
        return templates

    # ── Roboflow inference helper ─────────────────────────────────────────────

    def _troop_predict(self, image_path):
        """
        Run troop/tower detection via hosted Roboflow model.
        Returns list of {"class": str, "x": float, "y": float, "conf": float}.
        """
        result = self.rf_model.infer(image_path, model_id=TROOP_MODEL_ID)
        preds = []
        for p in result.get("predictions", []):
            preds.append({
                "class": p["class"],
                "x":     p["x"],           # pixel centre-x
                "y":     p["y"],           # pixel centre-y
                "conf":  p["confidence"],
            })
        return preds

    # ── Card detection via OpenCV template matching ───────────────────────────

    def detect_cards_in_hand(self):
        """
        Capture the 4 card slots and identify each card by comparing it against
        the reference templates in screenshots/card_*.png using template matching.

        Returns a list of 4 card name strings (e.g. "card_1", "card_2", …).
        Returns "Unknown" for any slot where no match exceeds the threshold.
        """
        try:
            card_paths = self.actions.capture_individual_cards()
            print("\nDetecting cards via template matching:")

            if not self.card_templates:
                print("No card templates loaded — returning Unknown for all slots.")
                return ["Unknown"] * self.num_cards

            cards      = []
            threshold  = 0.6   # tune if needed (0.0–1.0, higher = stricter)

            for idx, card_path in enumerate(card_paths):
                live_img = cv2.imread(card_path, cv2.IMREAD_GRAYSCALE)
                if live_img is None:
                    print(f"  Slot {idx+1}: could not read captured image → Unknown")
                    cards.append("Unknown")
                    continue

                best_score = -1.0
                best_name  = "Unknown"

                for slot_idx, tmpl in self.card_templates.items():
                    tmpl_img = tmpl["img"]

                    # Resize template to match live image if sizes differ
                    if tmpl_img.shape != live_img.shape:
                        tmpl_resized = cv2.resize(
                            tmpl_img,
                            (live_img.shape[1], live_img.shape[0])
                        )
                    else:
                        tmpl_resized = tmpl_img

                    result = cv2.matchTemplate(live_img, tmpl_resized, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(result)

                    if max_val > best_score:
                        best_score = max_val
                        best_name  = tmpl["name"]

                if best_score >= threshold:
                    print(f"  Slot {idx+1}: {best_name} (score={best_score:.2f})")
                    cards.append(best_name)
                else:
                    print(f"  Slot {idx+1}: Unknown (best score={best_score:.2f} < {threshold})")
                    cards.append("Unknown")

            return cards

        except Exception as e:
            print(f"Error in detect_cards_in_hand: {e}")
            return ["Unknown"] * self.num_cards

    # ── Environment API ───────────────────────────────────────────────────────

    def reset(self):
        time.sleep(3)
        self.game_over_flag = None
        self._endgame_thread_stop.clear()
        self._endgame_thread = threading.Thread(target=self._endgame_watcher, daemon=True)
        self._endgame_thread.start()
        self.prev_elixir              = None
        self.prev_enemy_presence      = None
        self.prev_enemy_princess_towers = self._count_enemy_princess_towers()
        self.match_over_detected      = False
        state = self._get_state()
        if state is None:
            state = np.zeros(self.state_size, dtype=np.float32)
        return state

    def close(self):
        self._endgame_thread_stop.set()
        if self._endgame_thread:
            self._endgame_thread.join()

    def step(self, action_index):
        # ── match-over check ──────────────────────────────────────────────────
        if (not self.match_over_detected
                and hasattr(self.actions, "detect_match_over")
                and self.actions.detect_match_over()):
            print("Match over detected — forcing no-op.")
            self.match_over_detected = True

        if self.match_over_detected:
            action_index = len(self.available_actions) - 1  # no-op

        # ── game-over check ───────────────────────────────────────────────────
        if self.game_over_flag:
            done   = True
            state  = self._get_state() or np.zeros(self.state_size, dtype=np.float32)
            reward = self._compute_reward(state)
            result = self.game_over_flag
            if result == "victory":
                reward += 100
                print("Victory detected — ending episode.")
            elif result == "defeat":
                reward -= 100
                print("Defeat detected — ending episode.")
            self.match_over_detected = False
            return state, reward, done

        # ── card detection ────────────────────────────────────────────────────
        self.current_cards = self.detect_cards_in_hand()
        print("\nCurrent cards in hand:", self.current_cards)

        # No-op if all cards are unknown or list is empty
        if not self.current_cards or all(c == "Unknown" for c in self.current_cards):
            print("All cards Unknown — skipping move.")
            pyautogui.moveTo(1611, 831, duration=0.2)
            pyautogui.click()
            state = self._get_state() or np.zeros(self.state_size, dtype=np.float32)
            return state, 0, False

        # ── execute action ────────────────────────────────────────────────────
        action = self.available_actions[action_index]
        card_index, x_frac, y_frac = action
        print(f"Action: card={card_index}, x={x_frac:.2f}, y={y_frac:.2f}")

        spell_penalty = 0

        if card_index != -1 and card_index < len(self.current_cards):
            card_name = self.current_cards[card_index]
            print(f"Playing {card_name}")
            x = int(x_frac * self.actions.WIDTH)  + self.actions.TOP_LEFT_X
            y = int(y_frac * self.actions.HEIGHT) + self.actions.TOP_LEFT_Y
            self.actions.card_play(x, y, card_index)
            time.sleep(1)

            # Spell penalty: punish casting into empty area
            if card_name in SPELL_CARDS:
                state = self._get_state() or np.zeros(self.state_size, dtype=np.float32)
                enemy_positions = []
                for i in range(1 + 2 * MAX_ALLIES, 1 + 2 * MAX_ALLIES + 2 * MAX_ENEMIES, 2):
                    ex, ey = state[i], state[i + 1]
                    if ex != 0.0 or ey != 0.0:
                        enemy_positions.append((
                            int(ex * self.actions.WIDTH),
                            int(ey * self.actions.HEIGHT)
                        ))
                radius = 100
                hit = any(
                    ((ex - x) ** 2 + (ey - y) ** 2) ** 0.5 < radius
                    for ex, ey in enemy_positions
                )
                if not hit:
                    spell_penalty = -5

        # ── princess tower reward ─────────────────────────────────────────────
        current_towers   = self._count_enemy_princess_towers()
        tower_reward     = 0
        if self.prev_enemy_princess_towers is not None:
            if current_towers < self.prev_enemy_princess_towers:
                tower_reward = 20
        self.prev_enemy_princess_towers = current_towers

        next_state = self._get_state() or np.zeros(self.state_size, dtype=np.float32)
        reward     = self._compute_reward(next_state) + spell_penalty + tower_reward
        return next_state, reward, False

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_state(self):
        """Capture screen and return state vector, or None if detection fails."""
        self.actions.capture_area(self.screenshot_path)
        elixir = self.actions.count_elixir()

        predictions = self._troop_predict(self.screenshot_path)
        if not predictions:
            print("WARNING: No troop predictions found.")
            return None

        print("Detected classes:", [p["class"] for p in predictions])

        TOWER_CLASSES = {
            "ally king tower", "ally princess tower",
            "enemy king tower", "enemy princess tower"
        }

        def nc(cls):
            return cls.strip().lower() if isinstance(cls, str) else ""

        allies  = [(p["x"], p["y"]) for p in predictions
                   if nc(p.get("class", "")) not in TOWER_CLASSES
                   and nc(p.get("class", "")).startswith("ally")]

        enemies = [(p["x"], p["y"]) for p in predictions
                   if nc(p.get("class", "")) not in TOWER_CLASSES
                   and nc(p.get("class", "")).startswith("enemy")]

        def normalize(units):
            return [(x / self.actions.WIDTH, y / self.actions.HEIGHT) for x, y in units]

        def pad(units, n):
            units = normalize(units)
            units += [(0.0, 0.0)] * (n - len(units))
            return units[:n]

        ally_flat  = [c for pos in pad(allies,  MAX_ALLIES)  for c in pos]
        enemy_flat = [c for pos in pad(enemies, MAX_ENEMIES) for c in pos]

        return np.array([elixir / 10.0] + ally_flat + enemy_flat, dtype=np.float32)

    def _compute_reward(self, state):
        if state is None:
            return 0
        elixir         = state[0] * 10
        enemy_ys       = state[1 + 2 * MAX_ALLIES::2]   # only y coords
        enemy_presence = float(sum(enemy_ys))
        reward         = -enemy_presence

        if self.prev_elixir is not None and self.prev_enemy_presence is not None:
            elixir_spent  = self.prev_elixir - elixir
            enemy_reduced = self.prev_enemy_presence - enemy_presence
            if elixir_spent > 0 and enemy_reduced > 0:
                reward += 2 * min(elixir_spent, enemy_reduced)

        self.prev_elixir         = elixir
        self.prev_enemy_presence = enemy_presence
        return reward

    def _count_enemy_princess_towers(self):
        self.actions.capture_area(self.screenshot_path)
        predictions = self._troop_predict(self.screenshot_path)
        return sum(
            1 for p in predictions
            if p.get("class", "").strip().lower() == "enemy princess tower"
        )

    def get_available_actions(self):
        actions = [
            [card, x / (self.grid_width - 1), y / (self.grid_height - 1)]
            for card in range(self.num_cards)
            for x in range(self.grid_width)
            for y in range(self.grid_height)
        ]
        actions.append([-1, 0, 0])  # no-op
        return actions

    def _endgame_watcher(self):
        while not self._endgame_thread_stop.is_set():
            result = self.actions.detect_game_end()
            if result:
                self.game_over_flag = result
                break
            time.sleep(0.5)