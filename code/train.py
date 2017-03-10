from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json

import tensorflow as tf

from nli_model import Statement, NLISystem
from os.path import join as pjoin

import numpy as np
import logging

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
tf.app.flags.DEFINE_float("optimize_hyperparameters", True, "Whether to optimize hyperparameters or not")

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

def load_dataset(tier, num_samples=None): # tier: 'train', 'dev', 'test'
  premises = []
  hypotheses = []
  goldlabels = []
  with open(pjoin(FLAGS.data_dir, tier + '.ids.premise')) as premise_file, \
      open(pjoin(FLAGS.data_dir, tier + '.ids.hypothesis')) as hypothesis_file, \
      open(pjoin(FLAGS.data_dir, tier + '.goldlabel')) as goldlabel_file:

      if num_samples:
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

def main(_):

    # Do what you need to load datasets from FLAGS.data_dir
    train_dataset = load_dataset('train', 10000)
    test_dataset = load_dataset('dev', 100)

    # Define paths
    embed_path = FLAGS.embed_path or pjoin("data", "snli", "glove.trimmed.{}.npz".format(FLAGS.embedding_size))
    vocab_path = FLAGS.vocab_path or pjoin(FLAGS.data_dir, "vocab.dat")

    # Get vocab and embeddings
    with np.load(embed_path) as embeddings_dict:
      embeddings = embeddings_dict['glove']
      vocab, rev_vocab = initialize_vocab(vocab_path)

      def run_model(lr, dropout_keep, reg_lambda):
        # Initalize the NLI System
        premise = Statement()
        hypothesis = Statement()
        nli = NLISystem(embeddings, premise, hypothesis, len(vocab), FLAGS.embedding_size, FLAGS.num_classes, lr, dropout_keep, reg_lambda)

        if not os.path.exists(FLAGS.log_dir):
          os.makedirs(FLAGS.log_dir)
          file_handler = logging.FileHandler(pjoin(FLAGS.log_dir, "log.txt"))
          logging.getLogger().addHandler(file_handler)

        # print(vars(FLAGS))
        with open(os.path.join(FLAGS.log_dir, "flags.json"), 'w') as fout:
          json.dump(FLAGS.__flags, fout)

        with tf.Session() as sess:
          initialize_model(sess, nli, FLAGS.load_train_dir)

          # Train the model
          epoch_number, train_accuracy, train_loss = nli.train(sess, train_dataset, rev_vocab, FLAGS.train_dir, FLAGS.batch_size)

          # Evaluate on the dev set
          test_accuracy, avg_test_loss, cm = nli.evaluate_prediction(sess, test_dataset)

          return (epoch_number, train_accuracy, train_loss, test_accuracy, avg_test_loss, cm)

      # RUN THE MODEL
      if not FLAGS.optimize_hyperparameters:
        run_model(FLAGS.lr, FLAGS.dropout_keep, FLAGS.reg_lambda)
      else:
        lr_range = np.array([10**lr_exp for lr_exp in range(-8, 0, 1)])
        dropout_range = np.arange(0.1, 1.1, 0.1) 
        reg_lambda_range = np.array([10**lr_exp for lr_exp in range(-4, 0, 1)]) 
        results_grid = np.empty(shape=[len(lr_range), len(dropout_range), len(reg_lambda)])
        for x, lr in enumerate(lr_range):
          for y, dropout_keep in enumerate(dropout_range):
            for z, reg_lambda in enumerate(reg_lambda_range):
              results_grid[x, y, z] = run_model(lr, dropout_keep, reg_lambda)

if __name__ == "__main__":
  tf.app.run()

