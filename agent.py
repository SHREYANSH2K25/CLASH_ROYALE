import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from pynput import keyboard
from datetime import datetime
import glob
import json
from env2 import ClashRoyaleEnv


class KeyboardController:
    def __init__(self):
        self.should_exit = False
         
        # .listener() defines what thread must execute when started
        self.listener = keyboard.Listener(on_press=self.on_press) # execute this function
        self.listener.start() # creates a thread

    def on_press(self, key):
        try:
            if key.char == 'q':
                print("\nShutdown requested")
                self.should_exit = True
        except AttributeError:
            pass  # Special key pressed
            
    def is_exit_requested(self):
        return self.should_exit


#Neural network
class DQN(nn.Module):
    def __init__(self, num_input, num_hidden, out):
        super().__init__()
        self.fc1 = nn.Linear(num_input, num_hidden) 
        self.fc2 = nn.Linear(num_hidden, out)
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        return self.fc2(x)


#Replay buffer
class ReplayMemory():
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)
    
    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)


#DQN Agent
class DQN_agn:

    def __init__(self, state_size, action_size, hidden_size=64):
        self.model = DQN(state_size, hidden_size, action_size)
        self.target_model = DQN(state_size, hidden_size, action_size)
        # sync target network weights at start
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.gamma = 0.95

    # writing to the hard drive
    # state_dict() -> python lib that maps each NN layer to its curr weights
    # torch.save() takes that dict and saves it into a file
    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.load_state_dict(torch.load(path, map_location=device))
        # CRUCIAL : function updates target model after set of steps
        self.target_model.load_state_dict(self.model.state_dict())


    @staticmethod
    def get_latest_model_path(models_dir="models"):
        model_files = glob.glob(os.path.join(models_dir, "model_*.pth"))
        if not model_files:
            return None
        model_files.sort()
        return model_files[-1]


def train():
    env = ClashRoyaleEnv()

    memory       = ReplayMemory(10000)
    batch_size   = 32
    episodes     = 50
    epsilon      = 1.0
    epsilon_min  = 0.01
    epsilon_decay = 0.997
    action_size  = env.action_size

    agent = DQN_agn(env.state_size, action_size)

    os.makedirs("models", exist_ok=True)

    latest_model = DQN_agn.get_latest_model_path("models")
    if latest_model:
        agent.load(latest_model)
        meta_path = latest_model.replace("model_", "meta_").replace(".pth", ".json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
                epsilon = meta.get("epsilon", 1.0)
            print(f"Epsilon loaded: {epsilon}")

    controller = KeyboardController()
    for ep in range(episodes):
        if controller.is_exit_requested():
            print("Training interrupted by user.")
            break
        # env.click_battle_start()
        # time.sleep(1)
        state = env.reset()

        print("reset")
        print(f"Episode {ep + 1} starting. Epsilon: {epsilon:.3f}")
        total_reward = 0
        done = False

        while not done:
            if controller.is_exit_requested():
                print("Training interrupted by user.")
                return 
            if random.random() < epsilon:
                action = random.randrange(action_size)
            else:
                with torch.no_grad():
                    q_values = agent.model(torch.FloatTensor(state).unsqueeze(0))
                action = q_values.argmax().item()
            # return

            next_state, reward, done = env.step(action)

            if next_state is None:
                continue

            memory.append((state, action, reward, next_state, done))

            total_reward += reward
            state = next_state

            if len(memory) < batch_size:
                continue

            batch = memory.sample(batch_size)
            for s, a, r, s2, d in batch:
                target = r
                if not d:
                    target += agent.gamma * torch.max(
                        agent.target_model(torch.tensor(s2,dtype=torch.float32))
                    ).item()

                target_f = agent.model(torch.FloatTensor(s)).clone().detach()
                target_f[a] = float(target)
                
                prediction = agent.model(torch.tensor(s, dtype=torch.float32))[a]

                target_tensor = torch.tensor(target, dtype=torch.float32)

                loss = agent.criterion(prediction, target_tensor)
                # prediction = agent.model(torch.FloatTensor(s))[a]
                # loss = agent.criterion(prediction, target_f[a])

                agent.optimizer.zero_grad()
                loss.backward()
                agent.optimizer.step()

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        print(f"Episode {ep + 1}: Total Reward = {total_reward:.2f}, Epsilon = {epsilon:.3f}")

        if ep % 10 == 0:
            agent.target_model.load_state_dict(agent.model.state_dict())
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join("models", f"model_{timestamp}.pth")
            torch.save(agent.model.state_dict(), model_path)
            with open(os.path.join("models", f"meta_{timestamp}.json"), "w") as f:
                json.dump({"epsilon": epsilon}, f)
            print(f"Model and epsilon saved to {model_path}")


if __name__ == "__main__":
    train()
    # print(5)