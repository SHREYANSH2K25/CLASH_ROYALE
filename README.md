# CLASH ROYALE AI Bot

An intelligent Clash Royale game automation bot using **Deep Q-Network (DQN)** reinforcement learning to autonomously play matches in Clash Royale. The bot detects game state using YOLO object detection and makes strategic card placements through learned policies.

---

## 🎮 Project Overview

This project implements an AI agent that learns to play **Clash Royale** (a real-time strategy card game) by:
- Capturing real-time game frames from BlueStacks emulator
- Detecting troop positions and available cards using YOLO models
- Making strategic decisions via a trained DQN neural network
- Executing actions (card placement, attacks) through automated mouse/keyboard control

The agent is trained using **reinforcement learning** (Q-learning) to maximize rewards based on game performance metrics like elixir management, tower damage, and match outcomes.

---

## 📂 Project Structure

```
CLASH_ROYALE/
├── agent.py              # DQN agent implementation & training loop
├── env2.py              # Game environment & state management
├── actions.py           # Screen capture & game interaction layer
├── data.py              # Game data utilities
├── elixir_detection.py  # Elixir counter detection
├── env.py               # Alternative environment interface
├── test.py              # Testing utilities
├── test_cards.py        # Card detection testing
├── requirements.txt     # Python dependencies
├── YOLO_MODEL/          # Pre-trained YOLO models for detection
│   ├── card.pt         # Card detection model
│   ├── troop.pt        # Troop detection model
│   └── troop6.pt       # Enhanced troop detection model
├── models/              # Trained DQN models (generated during training)
│   ├── model_*.pth     # Agent neural network weights
│   └── meta_*.json     # Training metadata
├── images/              # Training/reference images
├── check_images/        # Screenshot validation images
└── screenshots/         # Game screenshots for analysis
```

---

## 🔧 Key Components

### 1. **agent.py** - DQN Agent & Training

The core reinforcement learning implementation featuring:

#### **Classes:**
- **`KeyboardController`**: Manages graceful shutdown via 'Q' key press
  - Runs on separate thread using `pynput` library
  - Enables safe termination during training

- **`DQN`**: Neural network model (PyTorch)
  - Input: Game state (elixir + troop positions)
  - Hidden layer: 64 units with ReLU activation
  - Output: Q-values for each possible action
  - Architecture: `state_size → 64 → action_size`

- **`ReplayMemory`**: Experience buffer for training stability
  - Stores (state, action, reward, next_state, done) tuples
  - Max capacity: 10,000 transitions
  - Provides random sampling for mini-batch training

- **`DQN_agn`**: Training-ready agent wrapper
  - Maintains both main and target networks (for stable learning)
  - Implements model persistence (save/load functionality)
  - Auto-loads latest model from `models/` directory

#### **Training Loop (train() function):**
```python
- Episodes: 50 (configurable)
- Batch size: 32
- Epsilon (exploration): 1.0 → 0.01 (decays by 0.997 each episode)
- Learning rate: 0.001 (Adam optimizer)
- Discount factor (gamma): 0.95
```

#### **Features:**
- ✅ Model checkpointing with metadata tracking
- ✅ Epsilon-greedy exploration strategy
- ✅ Target network synchronization
- ✅ Automatic model recovery on restart

---

### 2. **env2.py** - Game Environment

Manages the complete game state and interfaces with detection models.

#### **Configuration Constants:**
```python
INFER_W, INFER_H = 640, 640      # YOLO inference resolution
CONF_THRES = 0.1                 # Detection confidence threshold
EMA_ALPHA = 0.4                  # State smoothing factor
MAX_ALLIES = 10                  # Max tracked ally troops
MAX_ENEMIES = 10                 # Max tracked enemy troops
```

#### **ClashRoyaleEnv Class:**

**Initialization:**
- Auto-detects BlueStacks window position
- Loads YOLO models (troop & card detection)
- Sets up action space (4 cards × grid positions)
- Initializes state smoothing (EMA filter)
- Spawns end-game detection thread

**State Management:**
```python
state_size = 1 + 2 * (MAX_ALLIES + MAX_ENEMIES)
           = 1 + 40 = 41 values
           
State = [elixir_count, 
         ally_x1, ally_y1, ally_x2, ally_y2, ..., 
         enemy_x1, enemy_y1, enemy_x2, enemy_y2, ...]
```

**Action Space:**
- 4 available card slots
- 18 × 28 grid positions for placement
- Total actions: `4 cards × 18 × 28 = 2,016`

**Key Methods:**
- `_capture_frame()`: In-memory screenshot caching
- `_run_yolo_inference()`: Detect troops and enemies
- `_extract_troop_info()`: Parse detection results
- `_detect_cards()`: Identify available cards in hand
- `_detect_elixir()`: Read current elixir count
- `_detect_game_over()`: Monitor match end conditions

**Reward System (tracks):**
- `prev_elixir`: Elixir efficiency
- `prev_enemy_presence`: Successful attacks
- `prev_enemy_princess_towers`: Tower destruction
- `match_over_detected`: Game end state

---

### 3. **actions.py** - Game Interaction Layer

Handles screen capture, window management, and automated game control.

#### **Initialization:**
```python
# Auto-detects BlueStacks window or accepts manual coordinates
Actions(TOP_LEFT_X, TOP_LEFT_Y, BOTTOM_RIGHT_X, BOTTOM_RIGHT_Y)
```

#### **Screen Capture:**
- **`capture_area()`**: Full emulator screenshot as numpy BGR array
- **`capture_card_area()`**: Card bar region (cards in hand)
- **`capture_individual_cards()`**: Splits card bar into 4 equal slots

#### **Card Mapping:**
```python
card_keys = {
    0: '1',  # First card slot → press '1'
    1: '2',  # Second card slot → press '2'
    2: '3',  # Third card slot → press '3'
    3: '4'   # Fourth card slot → press '4'
}
```

#### **Coordinate System:**
```
Card bar positioning (relative to game window):
- X offset: 24% from left
- Y offset: 82% from top
- Width: 72% of game width
- Height: 13% of game height
```

#### **Game Interaction:**
- Mouse click placement (via `pyautogui`)
- Keyboard card selection (1-4 keys)
- Window activation & focus management
- Cross-platform support (Windows/macOS/Linux detection)

---

## 📦 Dependencies

```
opencv-python==4.13.0.92      # Image processing & manipulation
ultralytics==8.4.35           # YOLO object detection
torch==2.11.0                 # Deep learning framework
numpy==2.4.4                  # Numerical computing
pyautogui==0.9.53             # Mouse & keyboard automation
pygetwindow==0.0.9            # Window detection for BlueStacks
pyscreeze==0.1.30             # Image matching & recognition
pynput==1.8.1                 # Keyboard input handling
python-dotenv==1.0.1          # Environment configuration
```

---

## 🚀 Getting Started

### Prerequisites
1. **BlueStacks Emulator** - Running Clash Royale
2. **Python 3.8+** - For runtime
3. **GPU (Optional)** - CUDA-enabled GPU for faster inference

### Installation

```bash
# 1. Clone/download the project
cd CLASH_ROYALE

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify GPU availability (optional)
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"
```

### Running the Agent

```bash
# Start training
python agent.py

# Press 'Q' to gracefully shutdown during training
# Models auto-save to models/ directory with timestamps
```

---

## 🧠 How It Works

### Training Pipeline

```
1. INITIALIZATION
   ├─ Load latest saved model (if exists)
   ├─ Initialize replay memory (10K capacity)
   └─ Setup DQN agent with target network

2. EPISODE LOOP (50 episodes)
   ├─ Reset game environment
   ├─ For each game step:
   │  ├─ Capture frame from BlueStacks
   │  ├─ Run YOLO inference (troops & cards)
   │  ├─ Extract game state (elixir, positions)
   │  ├─ Epsilon-greedy action selection
   │  │   ├─ Explore (random): probability ε
   │  │   └─ Exploit (trained): probability 1-ε
   │  ├─ Execute card placement
   │  ├─ Calculate reward
   │  ├─ Store transition in memory
   │  └─ Update network (if batch ready)
   │
   ├─ Sync target network (every N steps)
   ├─ Decay exploration rate ε
   └─ Save checkpoint (if new best)

3. MODEL CHECKPOINT
   ├─ Save model weights (.pth)
   ├─ Save training metadata (.json)
   │  └─ timestamp, episode, loss, reward
   └─ Cleanup old checkpoints
```

### Learning Algorithm (DQN)

```
For each mini-batch of 32 transitions:

1. Current Q-value: Q(s, a) = model(s)[a]
2. Target Q-value: 
   - If terminal: Y = reward
   - Else: Y = reward + γ × max Q'(s', a')
   
3. Loss: L = (Q(s,a) - Y)²
4. Backprop & optimize weights
5. Every N steps: target_model ← model
```

---

## 📊 State & Action Details

### Game State Representation (41 values)

| Index | Content | Range |
|-------|---------|-------|
| 0 | Elixir count | 0-10 |
| 1-20 | Ally troops (10 × x,y) | 0-18, 0-28 |
| 21-40 | Enemy troops (10 × x,y) | 0-18, 0-28 |

### Action Space (2,016 total actions)

- **Card Selection**: Which of 4 cards to play
- **Placement**: Where on the 18×28 grid to place
- **Formula**: `action = card_id × (18 × 28) + grid_position`

---

## 🎯 Training Hyperparameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Episodes | 50 | Training duration |
| Batch Size | 32 | Mini-batch for network update |
| Replay Memory | 10,000 | Experience buffer capacity |
| Learning Rate | 0.001 | Optimizer step size (Adam) |
| Discount (γ) | 0.95 | Future reward weighting |
| Epsilon Start | 1.0 | Initial exploration probability |
| Epsilon End | 0.01 | Final exploration minimum |
| Epsilon Decay | 0.997 | Per-episode decay rate |
| Hidden Layer | 64 | DQN network width |

---

## 📈 Performance Metrics

The agent learns through:
- **Reward signals** from game outcomes
- **State smoothing** (EMA filter) for stability
- **Target network** separation to prevent instability
- **Replay buffer** for efficient learning

---

## 🔍 Debugging & Development

### Testing
- `test.py` - General testing utilities
- `test_cards.py` - Card detection validation
- `elixir_detection.py` - Elixir counter validation

### Screenshots & Analysis
- `screenshots/` - Captured game states for analysis
- `check_images/` - Reference images (winner, OK button, etc.)

### Common Issues

| Issue | Solution |
|-------|----------|
| "BlueStacks window not found" | Ensure BlueStacks is open & not minimized |
| CUDA out of memory | Reduce batch size or use CPU |
| Low detection accuracy | Retrain YOLO models with more data |
| Training crashes | Press 'Q' and adjust hyperparameters |

---

## 🎮 Supported Card Types

```python
Card Detection: Knight, Archers, Minions, Goblins, Spear Goblins, 
                Musketeer, Giant, Mini P.E.K.K.A

Tower Detection: Ally King Tower, Ally Princess Tower, 
                 Enemy King Tower, Enemy Princess Tower
```

---

## 📝 Model Checkpointing

Models are saved with metadata:
```
models/
├── model_20260405_232759.pth      # Neural network weights
└── meta_20260405_232759.json      # Training info
                                     {
                                       "episode": 42,
                                       "timestamp": "2026-04-05 23:27:59",
                                       "avg_reward": 15.3,
                                       "loss": 0.0234
                                     }
```

---

## 🛣️ Future Enhancements

- [ ] Multi-agent training (team battles)
- [ ] Curriculum learning (progressive difficulty)
- [ ] PPO/A3C algorithms for better convergence
- [ ] Real device support (iOS/Android)
- [ ] Web UI for monitoring training
- [ ] Competitive ladder ranking

---

## ⚖️ License

This project is for educational purposes only.

---

**Last Updated**: April 2026  
**Project Status**: Active Development (Training Phase) 
