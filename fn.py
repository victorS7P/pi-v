import numpy as np

def to_int (bin):
  return sum(1<<i for i, b in enumerate(bin) if b)

def to_bin (i, s = 9):
  a = np.array([int(x) for x in bin(i)[2:]])
  a.resize(s)

  return a