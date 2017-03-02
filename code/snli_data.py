import gzip
import os
import re
import tarfile
import argparse

from tensorflow.python.platform import gfile
import numpy as np
from os.path import join as pjoin
from tqdm import *


_PAD = b"<pad>"
_UNK = b"<unk>"
_START_VOCAB = [_PAD, _UNK]

PAD_ID = 0
UNK_ID = 1


def setup_args():
    parser = argparse.ArgumentParser()
    code_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)))
    vocab_dir = os.path.join("data", "snli")
    glove_dir = os.path.join("data", "dwr")
    source_dir = os.path.join("data", "snli")
    parser.add_argument("--source_dir", default=source_dir)
    parser.add_argument("--glove_dir", default=glove_dir)
    parser.add_argument("--vocab_dir", default=vocab_dir)
    parser.add_argument("--glove_dim", default=100, type=int)
    return parser.parse_args()


def basic_tokenizer(sentence):
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(" ", space_separated_fragment))
    return [w for w in words if w]


'''
Loads stored vocab file and returns as two objects (same list, but tuple order reversed)
'''
def initialize_vocabulary(vocabulary_path):
    # map vocab to word embeddings
    if gfile.Exists(vocabulary_path):
        rev_vocab = [] # Reversed vocab
        with gfile.GFile(vocabulary_path, mode="r") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)


'''
Loads the main glove file and creates matrix of only the embeddings in our vocab 
Saves trimmed glove matrix
TODO -- Set default for size
'''
def process_glove(args, vocab_list, save_path, size):
    """
    :param vocab_list: [vocab]
    :return:
    """
    if not gfile.Exists(save_path + ".npz"):
        glove_path = os.path.join(args.glove_dir, "glove.6B.{}d.txt".format(args.glove_dim))
        glove = np.zeros((len(vocab_list), args.glove_dim))
        not_found = 0
        with open(glove_path, 'r') as fh:
            for line in tqdm(fh, total=size):
                array = line.lstrip().rstrip().split(" ")
                word = array[0]
                vector = list(map(float, array[1:]))
                if word in vocab_list:
                    idx = vocab_list.index(word)
                    glove[idx, :] = vector
                elif word.capitalize() in vocab_list:
                    idx = vocab_list.index(word.capitalize())
                    glove[idx, :] = vector
                elif word.lower() in vocab_list:
                    idx = vocab_list.index(word.lower())
                    glove[idx, :] = vector
                elif word.upper() in vocab_list:
                    idx = vocab_list.index(word.upper())
                    glove[idx, :] = vector
                else:
                    not_found += 1
        found = size - not_found
        print("{}/{} of word vocab have corresponding vectors in {}".format(found, len(vocab_list), glove_path))
        np.savez_compressed(save_path, glove=glove)
        print("saved trimmed glove matrix at: {}".format(save_path))


'''
Go through each set of sentence pairs in our data (train, dev, test...) to aggregate vocabulary. 
Store vocabulary.
'''
def create_vocabulary(vocabulary_path, data_paths, tokenizer=None):
    if not gfile.Exists(vocabulary_path):
        print("Creating vocabulary %s from data %s" % (vocabulary_path, str(data_paths)))
        vocab = {} # Word: Count
        for path in data_paths:
            with open(path, mode="rb") as f:
                counter = 0
                for line in f:
                    counter += 1
                    if counter % 100000 == 0:
                        print("processing line %d" % counter)
                    tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
                    for w in tokens:
                        if w in vocab:
                            vocab[w] += 1
                        else:
                            vocab[w] = 1
        vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True) # Add placeholder tokens and sort by count
        print("Vocabulary size: %d" % len(vocab_list))
        with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
            for w in vocab_list:
                vocab_file.write(w + b"\n")


'''
Given sentence, tokenizes into words and then into word indices in vocabulary
'''
def sentence_to_token_ids(sentence, vocabulary, tokenizer=None):
    if tokenizer:
        words = tokenizer(sentence)
    else:
        words = basic_tokenizer(sentence)
    return [vocabulary.get(w, UNK_ID) for w in words]


'''
For a data set, initializes a vocab, 
'''
def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None):
    if not gfile.Exists(target_path):
        print("Tokenizing data in %s" % data_path)
        vocab, _ = initialize_vocabulary(vocabulary_path)
        with gfile.GFile(data_path, mode="rb") as data_file:
            with gfile.GFile(target_path, mode="w") as tokens_file:
                counter = 0
                for line in data_file:
                    counter += 1
                    if counter % 5000 == 0:
                        print("tokenizing line %d" % counter)
                    token_ids = sentence_to_token_ids(line, vocab, tokenizer)
                    tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")


if __name__ == '__main__':
    # ======== Set up arguments and paths =======
    args = setup_args()
    vocab_path = pjoin(args.vocab_dir, "vocab.dat")

    train_path = pjoin(args.source_dir, "train")
    dev_path = pjoin(args.source_dir, "dev")
    test_path = pjoin(args.source_dir, 'test')

    # ======== Read data from JSON into separate files =======
    
    # TODO: TAKE OUT SENTENCES W/O VALID GOLD LABEL

    # TODO(Kenny): Parse data from json objects into separate files

    # ======== Create Vocabulary =======
    # Create the de facto vocabulary and store it in vocab.dat
    create_vocabulary(vocab_path,
                      [pjoin(args.source_dir, "train.premise"),
                       pjoin(args.source_dir, "train.hypothesis"),
                       pjoin(args.source_dir, "dev.premise"),
                       pjoin(args.source_dir, "dev.hypothesis"),
                       pjoin(args.source_dir, "test.premise"),
                       pjoin(args.source_dir, "test.hypothesis")])
    # Retrieve the vocab that was just created
    vocab, rev_vocab = initialize_vocabulary(pjoin(args.vocab_dir, "vocab.dat"))

    # ======== Trim Distributed Word Representation =======
    process_glove(args, rev_vocab, args.source_dir + "/glove.trimmed.{}".format(args.glove_dim))

    # ======== Create Dataset =========
    x_train_ids_path = train_path + ".ids.premise"
    y_train_ids_path = train_path + ".ids.hypothesis"
    data_to_token_ids(train_path + ".premise", x_train_ids_path, vocab_path)
    data_to_token_ids(train_path + ".hypothesis", y_train_ids_path, vocab_path)

    x_dev_ids_path = dev_path + ".ids.premise"
    y_dev_ids_path = dev_path + ".ids.hypothesis"
    data_to_token_ids(dev_path + ".context", x_dev_ids_path, vocab_path)
    data_to_token_ids(dev_path + ".question", y_dev_ids_path, vocab_path)

    x_test_ids_path = test_path + ".ids.premise"
    y_test_ids_path = test_path + ".ids.hypothesis"
    data_to_token_ids(test_path + ".context", x_test_ids_path, vocab_path)
    data_to_token_ids(test_path + ".question", y_test_ids_path, vocab_path)
