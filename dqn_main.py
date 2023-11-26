import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque
from itertools import count
import random 
# Assuming env.py is in the same directory and env.py is the filename
from env import WareHouseEnv

# Set the device to run on GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# Hyperparameters
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
LEARNING_RATE = 1e-3

# Replay Memory
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def put(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Q-Network
class MultiQNet(nn.Module):
    def __init__(self, state_dim, action_dims):
        super(MultiQNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.heads = nn.ModuleList([nn.Linear(128, action_dim) for action_dim in action_dims])

    def forward(self, x):
        # Flatten the input x if it is not already a flat vector
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return [head(x) for head in self.heads]
# DQN Agent for MultiDiscrete action spaces
class MultiDQNAgent:
    def __init__(self, state_dim, action_dims):
        self.steps_done = 0
        self.policy_net = MultiQNet(state_dim, action_dims).to(device)
        self.target_net = MultiQNet(state_dim, action_dims).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = ReplayBuffer(10000)
        self.action_dims = action_dims

    # def select_action(self, state):
    #     global steps_done
    #     sample = random.random()
    #     eps_threshold = EPS_END + (EPS_START - EPS_END) * \
    #         np.exp(-1. * self.steps_done / EPS_DECAY)
    #     self.steps_done += 1
    #     if sample > eps_threshold:
    #         with torch.no_grad():
    #             return self.policy_net(state).max(1)[1].view(1, 1)
    #     else:
    #         return torch.tensor([[random.randrange(self.action_dim)]], device=device, dtype=torch.long)
    def select_action(self, state):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # Call policy_net once here
                q_values = self.policy_net(state)
                # Use the output q_values to get the actions
                return [q_values[i].max(1)[1].view(1, 1) for i in range(len(self.action_dims))]
        else:
            # Random action for each dimension
            return [torch.tensor([[random.randrange(dim)]], device=device, dtype=torch.long) for dim in self.action_dims]



    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        
        # Filter out the transitions with None next_state
        non_final_transitions = [t for t in transitions if t.next_state is not None]
        non_final_next_states = torch.cat([t.next_state for t in non_final_transitions])
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
        
        state_batch = torch.cat([t.state for t in transitions if t.state is not None])
        action_batch = torch.cat([t.action for t in transitions if t.action is not None])
        reward_batch = torch.cat([t.reward for t in transitions if t.reward is not None])
        
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        state_action_values = torch.cat([self.policy_net(state_batch)[i].gather(1, action_batch[i].unsqueeze(-1)) for i in range(len(self.action_dims))], dim=1)

        # Compute expected values for non final states
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = torch.cat([self.target_net(non_final_next_states)[i].max(1)[0] for i in range(len(self.action_dims))], dim=1)

        # Compute the expected Q values
        expected_state_action_values = (next_state_values.unsqueeze(1) * GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()


    # Main loop
def main():
    env = WareHouseEnv()
    state_dim = np.prod(env.observation_space.shape)
    action_dims = env.action_space.nvec.tolist() # Convert to list
    agent = MultiDQNAgent(state_dim, action_dims)

    num_episodes = 500
    for i_episode in range(num_episodes):
        # Initialize the environment and state
        state = env.reset()
        # Flatten the state
        state = np.stack([state.flatten()])# , device=device, dtype=torch.float32)

        for t in count():
            # Select and perform an action
            actions = agent.select_action(state)
            # Convert actions to a format that the environment can understand
            actions_env = [action.item() for action in actions]
            next_state, reward, done, _ = env.step(actions_env)
            # Flatten the next state and wrap it in a tensor
            reward = torch.tensor([reward], device=device, dtype=torch.float32)

            if not done:
                next_state = torch.tensor([next_state], device=device, dtype=torch.float32)
            else:
                next_state = None

            # Store the transition in memory
            agent.memory.put(state, actions, next_state, reward, torch.tensor([done], device=device, dtype=torch.bool))

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            agent.optimize_model()

            if done:
                break

        # Update the target network
        if i_episode % TARGET_UPDATE == 0:
            agent.update_target_net()

if __name__ == '__main__':
    main()