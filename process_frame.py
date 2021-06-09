import cv2
import numpy as np

# Converte um frame 210x160x3 em um frame 80x80x1
def process_frame(frame, shape=(80, 80)):
  frame = frame.astype(np.uint8) 

  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  frame = frame[34:34+len(frame), :len(frame)]

  frame = cv2.resize(frame, shape, interpolation=cv2.INTER_NEAREST)
  frame = frame.reshape((*shape, 1))

  return frame