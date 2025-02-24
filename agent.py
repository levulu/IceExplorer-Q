import gym
import numpy as np
import random
import matplotlib.pyplot as plt
import time
from matplotlib.animation import FuncAnimation

reward_tracking = []

env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="rgb_array")

q_table = np.zeros([env.observation_space.n, env.action_space.n])

size = int(np.sqrt(env.observation_space.n))
max_test_steps = size * 4

alpha_init = 0.2
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.999
epsilon_min = 0.01
episodes = 20000

for episode in range(episodes):
    state = env.reset()[0]
    done = False
    total_reward = 0
    alpha = max(0.1, alpha_init * (0.998 ** episode))

    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_state, reward, done, _, _ = env.step(action)

        q_table[state, action] = q_table[state, action] + alpha * (
                reward + gamma * np.max(q_table[next_state]) - q_table[state, action]
        )

        state = next_state
        total_reward += reward

    if episode % 1000 == 0:
        reward_tracking.append(total_reward)
        success_rate = (sum(reward_tracking[-10:]) / 10) * 100
        print(f"Episode {episode} - Success Rate: {success_rate:.2f}%")

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

state = env.reset()[0]
done = False
print("\n Trained Agent's Test Run \n")

random_exploration = 0.3 if size > 6 else 0.0

step_count = 0
frames = []

while not done and step_count < max_test_steps:
    if random.uniform(0, 1) < random_exploration:
        action = env.action_space.sample()
    else:
        action = np.argmax(q_table[state])

    next_state, reward, done, _, _ = env.step(action)
    frame = env.render()
    frames.append(frame)
    print(f"Test State: {state} → Action: {action} → New State: {next_state} → Reward: {reward}")
    state = next_state
    step_count += 1

print("Test completed.")

success_count = 0
test_episodes = 100

for _ in range(test_episodes):
    state = env.reset()[0]
    done = False
    step_count = 0

    while not done and step_count < max_test_steps:
        if random.uniform(0, 1) < random_exploration:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_state, reward, done, _, _ = env.step(action)
        state = next_state

        if reward == 1:
            success_count += 1

        step_count += 1

success_rate = (success_count / test_episodes) * 100
print(f" Agent's Success Rate: {success_rate:.2f}%")

plt.plot(range(0, episodes, 1000), reward_tracking)
plt.xlabel("Training Episodes")
plt.ylabel("Total Reward")
plt.title("Reward Accumulation in Reinforcement Learning")
plt.grid()
plt.show()

print("\n Agent's Movements After Training (Animation) \n")

fig, ax = plt.subplots()
ax.axis("off")

def update(frame_index):
    ax.imshow(frames[frame_index])

ani = FuncAnimation(fig, update, frames=len(frames), interval=500)
plt.show()

env.close()
print(" Simulation completed, window closing.")
