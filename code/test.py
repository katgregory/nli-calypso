from tqdm import *
from os.path import join as pjoin

import numpy as np

WORDS = "three men standing on grass by the water looking at something on a table"
words = WORDS.split()
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
