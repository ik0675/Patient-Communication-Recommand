import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# Step 1: Define Environment
class PatientEngagementEnv:
    def __init__(self):
        # States: [Visits, Missed Appointments, Age Group, Last Engagement Response (pos. vs neg.)]
        self.state_space = [0, 0, 0, 0]  # Example: [2 visits, 1 missed, middle-aged, positive response]
        self.action_space = [0, 1, 2, 3]  # Actions: [No engagement, Email, SMS, Incentive (e.g. gift cards, discount coupons, free services)]
        self.max_visits = 10
        self.max_missed = 5

    def reset(self):
        self.state_space = [random.randint(0, 3),  # Visits in last year
                            random.randint(0, 2),  # Missed appointments
                            random.randint(0, 2),  # Age group
                            random.randint(0, 1)]  # Last response
        return np.array(self.state_space)

    def step(self, action):
        # Simulate patient's response to engagement
        visits, missed, age_group, last_response = self.state_space
        reward = 0
        done = False

        if action == 1:  # Email
            response = np.random.choice([0, 1], p=[0.6, 0.4])
        elif action == 2:  # SMS
            response = np.random.choice([0, 1], p=[0.4, 0.6])
        elif action == 3:  # Incentive
            response = np.random.choice([0, 1], p=[0.3, 0.7])
        else:  # No engagement
            response = 0
        
        # Came vs. Not came
        if response == 1:
            visits += 1
            reward = 1
        else:
            missed += 1
            reward = -1

        if missed > self.max_missed:
            reward = -5
            done = True

        self.state_space = [visits, missed, age_group, response]
        return np.array(self.state_space), reward, done

# Step 2: Define DQN Model
# A Neural Network predicts the Q-values (expected future rewards) for each action based on the current state.
# Architecture:
#   Input: Patient state
#   Hidden Layers: 2 layers with 64 neurons each, using ReLU activation.
#   Output: Q-values for each action (4 possible actions).
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

# Step 3: Train the Model
def train_dqn():
    env = PatientEngagementEnv()
    state_size = len(env.state_space)
    action_size = len(env.action_space)

    model = DQN(state_size, action_size)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Replay buffer
    replay_buffer = deque(maxlen=1000)
    batch_size = 32

    episodes = 1000
    gamma = 0.99  # Discount factor
    # Epsilon is the probability of taking a random action rather than choosing the action that maximizes the Q-value predicted by the model.
    # Typically starts high (e.g., 1.0, meaning complete exploration at the start) and decreases over time.
    epsilon = 1.0  # Exploration rate
    epsilon_decay = 0.995
    epsilon_min = 0.1

    for episode in range(episodes):
        state = env.reset()
        state = torch.FloatTensor(state)
        total_reward = 0

        for t in range(100):
            # Choose action
            if random.random() < epsilon:
                action = random.choice(env.action_space)
            else:
                q_values = model(state)
                action = torch.argmax(q_values).item()

            # Take action
            next_state, reward, done = env.step(action)
            next_state = torch.FloatTensor(next_state)

            # Store transition
            replay_buffer.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            # Train model
            # Bellman equation: 
            #   target_q_value = reward + gamma * max(next_q_value) * (1 - done)
            if len(replay_buffer) >= batch_size:
                minibatch = random.sample(replay_buffer, batch_size)
                states, actions, rewards, next_states, dones = zip(*minibatch)

                states = torch.stack(states)
                actions = torch.LongTensor(actions)
                rewards = torch.FloatTensor(rewards)
                next_states = torch.stack(next_states)
                dones = torch.FloatTensor(dones)

                q_values = model(states)
                q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze()

                next_q_values = model(next_states).max(1)[0]
                targets = rewards + gamma * next_q_values * (1 - dones)

                loss = criterion(q_values, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break

        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

    return model

# Step 4: Evaluate the Model
trained_model = train_dqn()
env = PatientEngagementEnv()

state = env.reset()
for _ in range(10):
    state = torch.FloatTensor(state)
    action = torch.argmax(trained_model(state)).item()
    next_state, reward, done = env.step(action)
    print(f"Action: {action}, Reward: {reward}, Next State: {next_state}")
    state = next_state
    if done:
        break