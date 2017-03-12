import cPickle as pickle
import os
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, axes3d
import numpy as np
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches


file_path = 'gpu-runs/run2/grid.p'
output_file_name = 'search'

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

lr = []
dropout_keep = []
reg_lambda = []
test_accuracy = []

keys = results_map.keys()
for key in keys:
  lr.append(key[0])
  dropout_keep.append(key[1])
  reg_lambda.append(key[2])
  test_accuracy.append(results_map[key][3])

def log_tick_formatter(val, pos=None):
    return "{:.2e}".format(10**val)


# sp = ax.scatter(np.log10(lr), dropout_keep, test_accuracy, c=reg_lambda, cmap=cm.coolwarm)
# sp = ax.scatter(np.log10(lr), dropout_keep, test_accuracy)

# PROCESS DATA FOR PLOTTING
lr = np.log10(lr)
graph_map = {}
for i in xrange(len(lr)):
  graph_map[(lr[i], dropout_keep[i], reg_lambda[i])] = test_accuracy[i]

X = np.unique(np.array(sorted(lr)))
Y = np.unique(np.array(sorted(dropout_keep)))
X, Y = np.meshgrid(X, Y)

# For each reg_lambda
Zs = []
for reg_lam in reg_lambda:
  Z = np.empty(np.shape(X))
  for i in xrange(np.shape(Z)[0]):
    for j in xrange(np.shape(Z)[1]):
      Z[i, j] = graph_map[(X[i, j], Y[i, j], reg_lam)]
  Zs.append(Z)

colors = ["red", "blue", "green", "yellow", "purple", "orange", "black"]
reg_lambda = np.array(np.unique(reg_lambda))

#### PLOTTING CODE ###########
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in xrange(len(reg_lambda)):
  ax.plot_wireframe(X, Y, Zs[i], color=colors[i])

ax.set_xlabel('lr')
# lr is on a log scale
ax.xaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
ax.set_ylabel('dropout_keep')
ax.set_zlabel('test_accuracy')

# CREATE LEGEND
patches = []
for i in xrange(len(reg_lambda)):
  patches.append(mpatches.Patch(color=colors[i], label=str(reg_lambda[i])))
plt.legend(handles=patches, title="Regularization Constant")
# Color is reg_lambda
# fig.colorbar(sp, shrink=0.5, aspect=5)

plt.title("Hyperparameter Grid Search")
plt.savefig('./data/hyperparams/' + output_file_name + '.png')
plt.show()

# ax = fig.gca(projection='3d')
# x, y = np.meshgrid(np.array(x), np.array(y))
# surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
# ax.set_zlim(0, 1)
# fig.colorbar(surf, shrink=0.5, aspect=5)
# plt.show()
