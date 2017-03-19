import cPickle as pickle
import os
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from prompter import yesno

file_path = './analysis.out'

premises, hypotheses, es, corr, goldlabels, predicteds = pickle.load(open(file_path, 'rb'))

plot_size = int(raw_input("How many to plot at once?\n"))
plot_wrong = yesno('Prompt only wrong plots?')
num_plotted = 0

num_batches = len(premises)

def num_to_name(num):
  return {
    0: "entailment",
    1: 'neutral',
    2: 'contradiction'
  }[num]

for i in range(num_batches):
  premise_batch = premises[i]
  hypothesis_batch = hypotheses[i]
  e_batch = es[i]
  correct_batch = corr[i]
  labels_batch = goldlabels[i]
  predicteds_batch = predicteds[i]

  batch_size = len(premise_batch)
  for j in range(batch_size):
    premise = premise_batch[j]
    hypothesis = hypothesis_batch[j]
    e = e_batch[j]
    correct = correct_batch[j]
    label = labels_batch[j]
    predicted = predicteds_batch[j]

    if correct and plot_wrong: continue

    p_len = len(premise)
    h_len = len(hypothesis)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    e = np.delete(e, np.s_[h_len:], 1)
    e = np.delete(e, np.s_[p_len:], 0)

    print premise, hypothesis

    predicted = num_to_name(predicted)
    label = num_to_name(np.argmax(label))
    print "Predicted: ", predicted
    print "Gold: ", label

    title = "Predicted: " + predicted + "; Gold: " + label

    cax = ax.matshow(e, interpolation='nearest', cmap=cm.BuGn)
    fig.colorbar(cax)

    ax.set_xticks([k for k in range(len(hypothesis))])
    ax.set_xticklabels(hypothesis)
    ax.set_yticks([k for k in range(len(premise))])
    ax.set_yticklabels(premise)

    fig.canvas.set_window_title(title)

    plt.tight_layout()

    plt.show()


    if num_plotted % plot_size == plot_size - 1:
      raw_input("Press enter for next graph...")

    num_plotted += 1

