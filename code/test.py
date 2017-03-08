from tqdm import *

WORDS = "three men standing on grass by the water looking at something on a table"
words = WORDS.split()
embeddings = [None] * len(words)

with open("data/dwr/glove.6B.100d.txt", 'r') as fh:
    for line in tqdm(fh):
        array = line.lstrip().rstrip().split(" ")
        word = array[0]
        if word in words:
        	embeddings[words.index(word)] = list(map(float, array[1:]))

for word in embeddings:
	print word
	print ""