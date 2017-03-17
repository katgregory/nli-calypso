import cPickle as pickle
import os
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, axes3d
import numpy as np
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from scipy.interpolate import griddata


file_path = 'data/hyperparams/grid.p'
output_file_name = 'search'

results_map = pickle.load(open(file_path, 'rb'))

# best_train_accuracy = (None, -1)
# best_train_loss = (None, 100000)
# best_test_accuracy = (None, -1)
# best_test_loss = (None, 10000)

# for key, value in results_map.items():
#   _, train_accuracy, train_loss, test_accuracy, test_loss, _ = value

#   if train_accuracy > best_train_accuracy[1]:
#     best_train_accuracy = (key, train_accuracy)

#   if min(train_loss) < best_train_loss[1]:
#     best_train_loss = (key, min(train_loss))

#   if test_accuracy > best_test_accuracy[1]:
#     best_test_accuracy = (key, test_accuracy)

#   if test_loss < best_test_loss[1]:
#     best_test_loss = (key, test_loss)

# TODO: Visualization

# def print_tuple(tup):
#   return "Learning Rate: " + str(tup[0]) + "\tDropout Keep Rate: " + str(tup[1])

# print("Best training accuracy: " + str(best_train_accuracy[1]) + " achieved by: \n" + print_tuple(best_train_accuracy[0]) + "\n") 
# print("Best training loss: " + str(best_train_loss[1]) + " achieved by: \n" + print_tuple(best_train_loss[0]) + "\n") 
# print("Best testing accuracy: " + str(best_test_accuracy[1]) + " achieved by: \n" + print_tuple(best_test_accuracy[0]) + "\n") 
# print("Best testing loss: " + str(best_test_loss[1]) + " achieved by: \n" + print_tuple(best_test_loss[0]) + "\n") 


GRAPH_LOSS = False

lr = []
dropout_keep = []
test_accuracy = []

keys = results_map.keys()
for key in keys:
  lr.append(key[0])
  dropout_keep.append(key[1])
  if GRAPH_LOSS:
    test_accuracy.append(results_map[key][4])
  else:
    test_accuracy.append(results_map[key][3])

# OVERWRITES DATA WITH SOMETHING FOR TESTING
test_accuracy = [lr[i] + dropout_keep[i] for i in xrange(len(test_accuracy))]

def log_tick_formatter(val, pos=None):
    return "{:.2e}".format(10**val)

# sp = ax.scatter(np.log10(lr), dropout_keep, test_accuracy, c=reg_lambda, cmap=cm.coolwarm)
# sp = ax.scatter(np.log10(lr), dropout_keep, test_accuracy)

######## INTERPOLATION ##################
points = np.random.rand(20, 2) * 10
values = [x*(1-x)*np.cos(4*np.pi*x) * np.sin(4*np.pi*y**2)**2 for x, y in points]
grid_x, grid_y = np.mgrid[0.00001:0.001:100j, 0.51:0.99:100j]
grid_z = griddata(points, values, (grid_x, grid_y), method='cubic')


#(Unnecessary?)
######### Create Map for Z-Axis ################
# graph_map = {}
# for i in xrange(len(lr)):
#   graph_map[(lr[i], dropout_keep[i])] = test_accuracy[i]

####### GRIDDING ########################
# X = np.unique(np.array(sorted(lr)))
# Y = np.unique(np.array(sorted(dropout_keep)))
# X, Y = np.meshgrid(X, Y)
# Z = np.empty(np.shape(X))
# for i in xrange(np.shape(Z)[0]):
#   for j in xrange(np.shape(Z)[1]):
#     if (X[i, j], Y[i, j]) in graph_map:
#       Z[i, j] = graph_map[(X[i, j], Y[i, j])]
# Z = [graph_map[(lr[i], dropout_keep[i])] for i in range(len(lr))]


#### PLOTTING CODE ###########
fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
ax = fig.gca(projection='3d')
# ax.plot_trisurf(lr, dropout_keep, Z, linewidth=0.2, antialiased=True)
surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
# fig.colorbar(surf, shrink=0.5, aspect=5) 
# ax.plot_wireframe(X, Y, Zs[i], color=colors[i])

# lr is on a log scale (not anymore)
# ax.xaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
ax.set_xlabel('lr')
ax.set_ylabel('dropout_keep')
ax.set_zlabel('test_accuracy')

# CREATE LEGEND
# patches = []
# for i in xrange(len(reg_lambda)):
  # patches.append(mpatches.Patch(color=colors[i], label=str(reg_lambda[i])))
# plt.legend(handles=patches, title="Regularization Constant")

plt.title("Hyperparameter Grid Search")
plt.savefig('./data/hyperparams/' + output_file_name + '.png')
plt.show()

# ax = fig.gca(projection='3d')
# x, y = np.meshgrid(np.array(x), np.array(y))
# surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
# ax.set_zlim(0, 1)
# fig.colorbar(surf, shrink=0.5, aspect=5)
# plt.show()
