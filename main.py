import time
import numpy as np
import retro

from fn import to_bin, to_int
from agent import DQNAgent
from process_frame import process_frame

# Build env (first level, right only)
env = retro.make(game='SpaceInvaders-Nes')
# env.action_space.sample()

# Parameters
# states = np.shape(process_frame(env.reset()))

env.reset()
action = env.action_space.sample()
while True:
  env.render()
  
  # action = to_bin(to_int(action))
  env.step(action)

  time.sleep(1/120)
  action = env.action_space.sample()

# Agent
# agent = DQNAgent(states=states, actions=np.array([x for x in range(512)]), max_memory=100000, double_q=True)
# # agent.replay(env, './models', 1, False)

# # Episodes
# episodes = 10000
# rewards = []

# # Timing
# start = time.time()
# step = 0

# # Main loop
# for e in range(episodes):
#     # Reset env
#     state = process_frame(env.reset())

#     # Reward
#     total_reward = 0
#     iteration = 0

#     # Play
#     while True:
#         # Show env
#         # env.render()

#         # Run agent
#         action = agent.run(state=state)

#         # Perform action
#         next_state, reward, done, info = env.step(action)

#         # Remember
#         agent.add(experience=(state, process_frame(next_state), to_int(action), reward, done, info))

#         # Replay
#         agent.learn()

#         # Total reward
#         total_reward += reward

#         # Update state
#         state = process_frame(next_state)

#         # Increment
#         iteration += 1

#         # If done break loop
#         if done:
#           break

#     # Rewards
#     rewards.append(total_reward / iteration)

#     # Print
#     if e % 100 == 0:
#         print('Episode {e} - '
#               'Frame {f} - '
#               'Frames/sec {fs} - '
#               'Epsilon {eps} - '
#               'Mean Reward {r}'.format(e=e,
#                                        f=agent.step,
#                                        fs=np.round((agent.step - step) / (time.time() - start)),
#                                        eps=np.round(agent.eps, 4),
#                                        r=np.mean(rewards[-100:])))
#         start = time.time()
#         step = agent.step

# # Save rewards
# np.save('rewards.npy', rewards)
