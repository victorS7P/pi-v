import time
import numpy as np

from utils import to_bin, to_int

from env import Env, filtered_actions
from agent import Agent

env = Env()
state = env.reset()

# Parameters
state_shape = np.shape(env.process_frame(state))

while True:
  # Agent
  agent = Agent(state_shape=state_shape, actions=filtered_actions)
  # agent.replay(env, './models', 1, False)

  # Episodes
  episodes = 10000
  rewards = []

  # Timing
  start = time.time()
  step = 0

  # Main loop
  for e in range(episodes):
    # Reset env
    state = env.process_frame(env.reset())

    # Reward
    total_reward = 0
    iteration = 0

    # Play
    while True:
      # Show env
      env.render()

      # Run agent
      action = agent.run(state=state)

      # Perform action
      next_state, reward, done, info = env.step(action)

      # Remember
      agent.add(experience=(state, env.process_frame(next_state), to_int(action), reward, done, info))

      # Replay
      agent.learn()

      # Total reward
      total_reward += reward

      # Update state
      state = env.process_frame(next_state)

      # Increment
      iteration += 1

      # If done break loop
      if done or info['lives'] < 2:
        break

    # Rewards
    rewards.append(total_reward / iteration)

    # Print
    if e % 100 == 0:
        print(
          'Episode {e:5f} - '
          'Frame {f:10f} - '
          'Frames/sec {fs:5.2f} - '
          'Epsilon {eps:.5f} - '
          'Mean Reward {r:.5f}'.format(
            e=e,
            f=agent.step,
            fs=np.round((agent.step - step) / (time.time() - start)),
            eps=np.round(agent.eps, 4),
            r=np.mean(rewards[-100:])
          )
        )

        start = time.time()
        step = agent.step

  # Save rewards
  np.save('rewards.npy', rewards)
