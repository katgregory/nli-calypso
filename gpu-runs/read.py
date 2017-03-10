import numpy as np
import pickle

with open("grid.p") as f:
  x = pickle.load(f)
  print x
