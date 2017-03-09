from tqdm import *
from os.path import join as pjoin

import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from tensorflow.python.platform import gfile

def printGloveEmbeddings(sentence):
	words = sentence.split()
	embeddings = [None] * len(words)

	# with open("data/dwr/glove.6B.100d.txt", 'r') as fh:
	#     for line in tqdm(fh):
	#         array = line.lstrip().rstrip().split(" ")
	#         word = array[0]
	#         if word in words:
	#         	embeddings[words.index(word)] = list(map(float, array[1:]))

	embed_path = pjoin("data", "snli", "glove.trimmed.{}.npz".format(100))

	with np.load(embed_path) as embeddings_dict:
	  embeddings = embeddings_dict['glove']
	  print embeddings[2]           # a
	  print embeddings[3]           # A
	  print embeddings[6]           # the
	  print embeddings[13]          # The
	  print embeddings[40000]          # The

# for word in embeddings:
# 	print word
# 	print ""

# Generates toy data to test model
def generateToyData():
    toy_data_dir = "data/toy/"
    X, y = make_blobs(n_samples=1000, centers=3, n_features=1, random_state=0)
    with gfile.GFile(toy_data_dir + "train.premise", mode="w") as input_file:
        for line in X:
            input_file.write(str(line[0]) + "\n")
    with gfile.GFile(toy_data_dir + "train.goldlabel", mode="w") as output_file:
        for label in y:
            output_file.write(str(label) + "\n")

    X, y = make_blobs(n_samples=100, centers=3, n_features=1, random_state=0)
    with gfile.GFile(toy_data_dir + "test.premise", mode="w") as input_file:
        for line in X:
            input_file.write(str(line[0]) + "\n")
    with gfile.GFile(toy_data_dir + "test.goldlabel", mode="w") as output_file:
        for label in y:
            output_file.write(str(label) + "\n")

generateToyData()
