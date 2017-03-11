from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import json

import tensorflow as tf

from nli_model import Statement, NLISystem
from os.path import join as pjoin

import numpy as np
import logging
import cPickle as pickle

logging.basicConfig(level=logging.INFO)

# HYPERPARAMETERS
tf.app.flags.DEFINE_float("lr", 0.0001, "Learning rate.")
tf.app.flags.DEFINE_float("dropout_keep", 0.8, "Keep_prob")
tf.app.flags.DEFINE_float("reg_lambda", 0.01, "Regularization")

tf.app.flags.DEFINE_integer("batch_size", 64, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("epochs", 10, "Number of epochs to train.")

tf.app.flags.DEFINE_float("max_gradient_norm", 10.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("state_size", 200, "Size of each model layer.")
tf.app.flags.DEFINE_integer("output_size", 3, "The output size of your model.")
tf.app.flags.DEFINE_integer("embedding_size", 300, "Size of the pretrained vocabulary.")
tf.app.flags.DEFINE_string("data_dir", "data/snli", "snli directory (default ./data/snli)")
tf.app.flags.DEFINE_string("train_dir", "train", "Training directory to save the model parameters (default: ./train).")
tf.app.flags.DEFINE_string("load_train_dir", "", "Training directory to load model parameters from to resume training (default: {train_dir}).")
tf.app.flags.DEFINE_string("log_dir", "log", "Path to store log and flag files (default: ./log)")
tf.app.flags.DEFINE_string("optimizer", "adam", "adam / sgd")
tf.app.flags.DEFINE_integer("print_every", 1, "How many iterations to do per print.")
tf.app.flags.DEFINE_integer("keep", 0, "How many checkpoints to keep, 0 indicates keep all.")
tf.app.flags.DEFINE_string("vocab_path", "data/snli/vocab.dat", "Path to vocab file (default: ./data/snli/vocab.dat)")
tf.app.flags.DEFINE_string("embed_path", "", "Path to the trimmed GLoVe embedding (default: ./data/snli/glove.trimmed.{embedding_size}.npz)")
tf.app.flags.DEFINE_float("num_classes", 3, "Neutral, Entailment, Contradiction")
tf.app.flags.DEFINE_string("hyperparameter_grid_search_file", "data/hyperparams/grid.p", "Stores pickle file of search results")

FLAGS = tf.app.flags.FLAGS

def initialize_model(session, model, train_dir):
  ckpt = tf.train.get_checkpoint_state(train_dir)
  v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
  if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
      logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
      model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
      logging.info("Created model with fresh parameters.")
      session.run(tf.global_variables_initializer())
      logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
  return model


def initialize_vocab(vocab_path):
    if tf.gfile.Exists(vocab_path):
      rev_vocab = []
      with tf.gfile.GFile(vocab_path, mode="rb") as f:
        rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_path)

def convert_to_one_hot(label):
  if label == 'entailment':
    return np.array([1, 0, 0])
  elif label == 'neutral':
    return np.array([0, 1, 0])
  elif label == 'contradiction':
    return np.array([0, 0, 1])
  print ('failed to convert: ' + str(label))
  return 5/0

def load_dataset(tier, num_samples=-1): # tier: 'train', 'dev', 'test'
  premises = []
  hypotheses = []
  goldlabels = []
  with open(pjoin(FLAGS.data_dir, tier + '.ids.premise')) as premise_file, \
      open(pjoin(FLAGS.data_dir, tier + '.ids.hypothesis')) as hypothesis_file, \
      open(pjoin(FLAGS.data_dir, tier + '.goldlabel')) as goldlabel_file:

      if num_samples >= 0:
        for i in xrange(num_samples):
          premises.append(premise_file.readline().strip())
          hypotheses.append(hypothesis_file.readline().strip())
          goldlabels.append(convert_to_one_hot(goldlabel_file.readline().strip()))
      else:
        for line in premise_file:  
          premises.append(line.strip()) 
        for line in hypothesis_file:
          hypotheses.append(line.strip())
        for line in goldlabel_file:
          goldlabels.append(convert_to_one_hot(line.strip()))
      return (premises, hypotheses, goldlabels)

def run_model(embeddings, train_dataset, eval_dataset, vocab, rev_vocab, lr, dropout_keep, reg_lambda):

  # Reset every time. TODO: we should be using the same graph
  tf.reset_default_graph()

  # Initalize the NLI System
  nli = NLISystem(embeddings, FLAGS.num_classes, lr, dropout_keep, reg_lambda)

  if not os.path.exists(FLAGS.log_dir):
    os.makedirs(FLAGS.log_dir)
    file_handler = logging.FileHandler(pjoin(FLAGS.log_dir, "log.txt"))
    logging.getLogger().addHandler(file_handler)

  with open(os.path.join(FLAGS.log_dir, "flags.json"), 'w') as fout:
    json.dump(FLAGS.__flags, fout)

  with tf.Session() as sess:
    initialize_model(sess, nli, FLAGS.load_train_dir)

    # Train and evaluate the model
    epoch_number, train_accuracy, train_loss = nli.train(sess, train_dataset, rev_vocab, FLAGS.train_dir, FLAGS.batch_size)
    test_accuracy, avg_test_loss, cm = nli.evaluate_prediction(sess, FLAGS.batch_size, eval_dataset)
    return (epoch_number, train_accuracy, train_loss, test_accuracy, avg_test_loss, cm)

def validate_model(embeddings, train_dataset, eval_dataset, vocab, rev_vocab):
  lr_range = np.array([10**lr_exp for lr_exp in range(-8, -2, 1)])
  dropout_range = np.arange(0.1, 1.1, 0.1) 
  reg_lambda_range = np.array([10**lr_exp for lr_exp in range(-4, -1, 1)]) 

  results_map = {}
  best_train_accuracy = 0
  best_test_accuracy = 0
  for reg_lambda in reg_lambda_range:
    for dropout_keep in dropout_range:
      for lr in lr_range:
        print("########################################################")
        print("\nRUNNING TRIAL: ", "\tlr:", lr, "\tdropout:", dropout_keep, "\treg_lambda:", reg_lambda, "\n")
        idx_tup = (lr, dropout_keep, reg_lambda)
        results_map[idx_tup] = run_model(embeddings, train_dataset, eval_dataset, vocab, rev_vocab,
                                         lr, dropout_keep, reg_lambda)
        pickle.dump(results_map, open(FLAGS.hyperparameter_grid_search_file, "wb"))
        print("############################")
        print("TRIAL RESULTS: ", "\tlr:", lr, "\tdropout:", dropout_keep, "\treg_lambda:", reg_lambda, "\n")
        print("\tACCURACY: \ttrain:", results_map[idx_tup][1], "\ttest:", results_map[idx_tup][3])
        if results_map[idx_tup][1] > best_train_accuracy:
          best_train_accuracy = results_map[idx_tup][1]
          print("\t\tNew best TRAIN")
        if results_map[idx_tup][3] > best_test_accuracy:
          best_test_accuracy = results_map[idx_tup][3]
          print("\t\tNew best TEST")
        print("########################################################")

def main(purpose, num_train, num_eval, **hyperparams):

    # SET RANDOM SEED
    np.random.seed(244)

    # Load the two pertinent datasets
    train_dataset = load_dataset('train', num_train)
    if purpose == 'test':
      eval_dataset = load_dataset('test', num_eval)
    else:
      eval_dataset = load_dataset('dev', num_eval)

    # Define paths
    embed_path = FLAGS.embed_path or pjoin("data", "snli", "glove.trimmed.{}.npz".format(FLAGS.embedding_size))
    vocab_path = FLAGS.vocab_path or pjoin(FLAGS.data_dir, "vocab.dat")

    # Get vocab and embeddings
    with np.load(embed_path) as embeddings_dict:
      embeddings = embeddings_dict['glove']
      vocab, rev_vocab = initialize_vocab(vocab_path)

      if purpose == 'dev' or purpose == 'test':
        run_model(embeddings, train_dataset, eval_dataset, vocab, rev_vocab, hyperparams.get('lr'), 
                                                                 hyperparams.get('dropout_keep'), 
                                                                 hyperparams.get('reg_lambda')) 
      else: # purpose = 'validate'
        validate_model(embeddings, train_dataset, eval_dataset, vocab, rev_vocab)

################### ARGUMENT PARSER ##########################################
parser = argparse.ArgumentParser()
parser.add_argument('command', choices=['run', 'validation'])
parser.add_argument('--num_train', default=10000)
parser.add_argument('--dev', action='store_true')
parser.add_argument('--num_dev', default=1000)
parser.add_argument('--test', action='store_true')
parser.add_argument('--num_test', default=1000)
parser.add_argument('--lr', default=FLAGS.lr)
parser.add_argument('--dropout_keep', default=FLAGS.dropout_keep)
parser.add_argument('--reg_lambda', default=FLAGS.reg_lambda)
args = parser.parse_args()
if args.command == 'run':
  if (args.dev and args.test) or (not args.dev and not args.test):
    parser.error('Must set exactly one of: --dev or --test')

if __name__ == "__main__":
  if args.command == 'run' and args.dev:
    main('dev', int(args.num_train), int(args.num_dev), lr=float(args.lr), dropout_keep=float(args.dropout_keep), reg_lambda=float(args.reg_lambda))
  elif args.command == 'run' and args.test:
    main('test', int(args.num_train), int(args.num_test), lr=float(args.lr), dropout_keep=float(args.dropout_keep), reg_lambda=float(args.reg_lambda))
  elif args.command == 'validation':
    main('validate', int(args.num_train), int(args.num_dev))
  assert("Somehow flags are mismatched")







