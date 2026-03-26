import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import gymnasium as gym
from collections import deque
import matplotlib.pyplot as plt
from pynput import keyboard
from datetime import datetime
import glob 

# import environment

class KeyboardController:
    def __init__(self):
        self.should_exit = False
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()

    def on_press(self, key):
        try:
            if key.char == 'q':
                print("\nShutdown requested")
                self.should_exit = True
        except AttributeError:
            pass  # Special key pressed
            
    def is_exit_requested(self):
        return self.should_exit



class DQN(nn.Module):
    def __init__(self, num_input, num_hidden, out):
        super().__init__()
        self.fc1 = nn.Linear(num_input,64)
        self.fc2 = nn.Linear(64,out)
    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        return self.fc2(x)



class ReplayMemory():
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)
    
    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)



class DQN_agn:

    def __init__(self,state_size, action_size):
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        # self.memory = deque(maxlen=10000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.997
        self.action_size = action_size
    
    
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    # remember
    def remember(self, s, a, r, s2, done):
        self.memory.append((s, a, r, s2, done))
    
    
    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        return q_values.argmax().item()

    # looks into folder models
    def get_latest_model_path(models_dir="models"):
        # build a path like model_*.pth and glob finds all matching files
        # and now model_files have a list with all matched files 
        # like model_files = ["models/model_20240301.pth",]
        model_files = glob.glob(os.path.join(models_dir, "model_*.pth"))
        if not model_files:
            return None
        model_files.sort()  # Lexicographical sort works for timestamps
        # sort time wise and take the newest one
        return model_files[-1]


        
    def train():
        env = ClashRoyaleEnv()
        agent = DQNAgent(env.state_size, env.action_size)

        # Ensure models directory exists
        os.makedirs("models", exist_ok=True)

        # Load latest model if available
        latest_model = get_latest_model_path("models")
        if latest_model:
            agent.load(os.path.basename(latest_model))
            # Load epsilon
            meta_path = latest_model.replace("model_", "meta_").replace(".pth", ".json")
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                    agent.epsilon = meta.get("epsilon", 1.0)
                print(f"Epsilon loaded: {agent.epsilon}")

        controller = KeyboardController()
        episodes = 10000
        batch_size = 32

        for ep in range(episodes):
            if controller.is_exit_requested():
                print("Training interrupted by user.")
                break

            state = env.reset()
            print(f"Episode {ep + 1} starting. Epsilon: {agent.epsilon:.3f}")  # <-- Add this line
            total_reward = 0
            done = False
            while not done:
                action = agent.act(state)
                next_state, reward, done = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                agent.replay(batch_size)
                state = next_state
                total_reward += reward
            print(f"Episode {ep + 1}: Total Reward = {total_reward:.2f}, Epsilon = {agent.epsilon:.3f}")

            if ep % 10 == 0:
                agent.update_target_model()
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_path = os.path.join("models", f"model_{timestamp}.pth")
                torch.save(agent.model.state_dict(), model_path)
                with open(os.path.join("models", f"meta_{timestamp}.json"), "w") as f:
                    json.dump({"epsilon": agent.epsilon}, f)
                print(f"Model and epsilon saved to {model_path}")

if __name__ == "__main__":
    train()


