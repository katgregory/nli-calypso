import cPickle as pickle
import os
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


file_path = 'gpu-runs/run2/grid.p'

results_map = pickle.load(open(file_path, 'rb'))

best_train_accuracy = (None, -1)
best_train_loss = (None, 100000)
best_test_accuracy = (None, -1)
best_test_loss = (None, 10000)

for key, value in results_map.items():
  _, train_accuracy, train_loss, test_accuracy, test_loss, _ = value

  if train_accuracy > best_train_accuracy[1]:
    best_train_accuracy = (key, train_accuracy)

  if train_loss < best_train_loss[1]:
    best_train_loss = (key, train_loss)

  if test_accuracy > best_test_accuracy[1]:
    best_test_accuracy = (key, test_accuracy)

  if test_loss < best_test_loss[1]:
    best_test_loss = (key, test_loss)

# TODO: Visualization

def print_tuple(tup):
  return "Learning Rate: " + str(key[0]) + "\tDropout Rate: " + str(key[1]) + "\tRegularization lambda: " + str(key[2])

print("Best training accuracy: " + str(best_train_accuracy[1]) + " achieved by: \n" + print_tuple(best_train_accuracy[0]) + "\n") 
print("Best training loss: " + str(best_train_loss[1]) + " achieved by: \n" + print_tuple(best_train_loss[0]) + "\n") 
print("Best testing accuracy: " + str(best_test_accuracy[1]) + " achieved by: \n" + print_tuple(best_test_accuracy[0]) + "\n") 
print("Best testing loss: " + str(best_test_loss[1]) + " achieved by: \n" + print_tuple(best_test_loss[0]) + "\n") 


fig = plt.figure()
ax = fig.gca(projection='3d')

keys = results_map.keys()
x = []
y = []
z = []
for key in keys:
  if key[2] == 0.01:
    x.append(key[0])
    y.append(key[1])
    z.append(results_map[key][3])

x, y = np.meshgrid(np.array(x), np.array(y))

surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.set_zlim(0, 1)

fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
