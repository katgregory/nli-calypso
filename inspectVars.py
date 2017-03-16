import cPickle as pickle
import numpy as np


indexes = []
names = ['W1', 'b1', 'mul1', 'r1', 'W2', 'b2', 'mul2', 'r2']
with open("ff_vars") as f:
  for i, name in enumerate(names):
    var = pickle.load(f)
    print("P " + names[i] + ': ' + len(np.argwhere(np.isnan(varp))) + " out of "  + var.shape())

  for i, name in enumerate(names):
    var = pickle.load(f)
    print("H " + names[i] + ': ' + len(np.argwhere(np.isnan(varp))) + " out of " + var.shape())



