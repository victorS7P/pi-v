import random
import retro
import numpy
import cv2

from utils import to_bin

# 0 = tiro
# 6 = esquerda
# 7 = direita
filtered_actions = [to_bin(x, 3) for x in range(0, 8)]

class Env():
  def __init__ (self):
    self.env = retro.make(game='SpaceInvaders-Nes')

  def reset (self):
    return self.env.reset()

  def filter_action (self, action):
    return [action[0], action[6], action[7]]

  def expand_action (self, filtered_action):
    return [filtered_action[0], 0, 0, 0, 0, 0, filtered_action[1], filtered_action[2], 0, 0]

  def step (self, filtered_action):
    return self.env.step(self.expand_action(filtered_action))

  def random (self):
    return filtered_actions[random.randint(0, 7)]
  
  def render (self):
    return self.env.render()

  def pygame_state (self, state):
    return numpy.flip(numpy.rot90(state), 0)

  def process_frame (self, state):
    # (224, 240, 3) >> (95, 95, 1)
    frame = state.astype(numpy.uint8)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # cortamos o lado direito da tela
    frame = frame[20:len(frame)-13, 0:len(frame[0])-50]

    frame = cv2.resize(frame, (95, 95), interpolation=cv2.INTER_NEAREST)
    frame = numpy.reshape(frame, (95, 95, 1))

    frame = frame/255
    return frame