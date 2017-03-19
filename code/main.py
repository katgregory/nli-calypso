from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import json

import tensorflow as tf

from nli_model import NLISystem
from os.path import join as pjoin

import numpy as np
import logging
import cPickle as pickle

logging.basicConfig(level=logging.INFO)

# COMMAND LINE ARGUMENTS
tf.app.flags.DEFINE_bool("validation", False, "Tells us to perform Hyperparam validation")
tf.app.flags.DEFINE_bool("dev", False, "")
tf.app.flags.DEFINE_bool("test", False, "")
tf.app.flags.DEFINE_integer("num_train", 10000, "")
tf.app.flags.DEFINE_integer("num_dev", 1000, "")
tf.app.flags.DEFINE_integer("num_test", 1000, "")
tf.app.flags.DEFINE_bool("bucket", True, "")
tf.app.flags.DEFINE_string("stmt_processor", "bilstm", "How to process statements. Options: 'bow', 'lstm', 'bilstm'")
tf.app.flags.DEFINE_bool("infer_embeddings", False, "Include embeddings in inference step")
tf.app.flags.DEFINE_bool("train_embed", True, "Train the embeddings")
tf.app.flags.DEFINE_string("analysis_path", None, "Analysis output file")
tf.app.flags.DEFINE_string("restore_path", None, "Path from which to restore params")
tf.app.flags.DEFINE_bool("pool_merge", True, "Use max pool and average to merge.")
tf.app.flags.DEFINE_integer("n_bilstm_layers", 1, "Number of layers in the stacked bidirectional LSTM")
tf.app.flags.DEFINE_integer("max_grad_norm", -1, "For clipping")

# TYPES OF ATTENTION
tf.app.flags.DEFINE_bool("attentive_matching", False, "Chen's attention")
tf.app.flags.DEFINE_bool("weight_attention", False, "Adds weight multiplication to attention calculation")
tf.app.flags.DEFINE_bool("max_attentive_matching", False, "From Wang et al '17")
tf.app.flags.DEFINE_bool("full_matching", False, "From Wang et al '17")
tf.app.flags.DEFINE_bool("maxpool_matching", False, "From Wang et al '17")

# HYPERPARAMETERS
tf.app.flags.DEFINE_float("lr", 0.0004, "Learning rate.")
tf.app.flags.DEFINE_float("dropout_keep", 0.5, "Keep_prob")
tf.app.flags.DEFINE_float("reg_lambda", -1, "Regularization")

tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("epochs", 10, "Number of epochs to train.")

tf.app.flags.DEFINE_float("max_gradient_norm", 10.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("ff_hidden_size", 300, "Size of each model layer.")
tf.app.flags.DEFINE_integer("stmt_hidden_size", 100, "Size of hidden layer between LSTMs and FF when no attention.")
tf.app.flags.DEFINE_integer("lstm_hidden_size", 300, "Size of hidden layers in LSTM.")
tf.app.flags.DEFINE_integer("output_size", 3, "The output size of your model.")
tf.app.flags.DEFINE_integer("embedding_size", 300, "Size of the pretrained vocabulary.")
tf.app.flags.DEFINE_string("data_dir", "data/snli", "snli directory (default ./data/snli)")
tf.app.flags.DEFINE_string("train_dir", "train_params", "Training directory to save the model parameters")
tf.app.flags.DEFINE_string("validation_dir", "validation_params", "Validation directory to save the model parameters")
tf.app.flags.DEFINE_string("log_dir", "log", "Path to store log and flag files (default: ./log)")
tf.app.flags.DEFINE_string("tboard_path", None, "Path to store tensorboard files (default: None)")
tf.app.flags.DEFINE_string("optimizer", "adam", "adam / sgd")
tf.app.flags.DEFINE_integer("print_every", 1, "How many iterations to do per print.")
tf.app.flags.DEFINE_integer("keep", 0, "How many checkpoints to keep, 0 indicates keep all.")
tf.app.flags.DEFINE_string("vocab_path", "data/snli/vocab.dat", "Path to vocab file (default: ./data/snli/vocab.dat)")
tf.app.flags.DEFINE_string("embed_path", "", "Path to the trimmed GLoVe embedding (default: ./data/snli/glove.trimmed.{embedding_size}.npz)")
tf.app.flags.DEFINE_float("num_classes", 3, "Neutral, Entailment, Contradiction")
tf.app.flags.DEFINE_integer("ff_num_layers", 2, "Number of layers in final FF network")
tf.app.flags.DEFINE_string("hyperparameter_grid_search_file", "data/hyperparams/grid.p", "Stores pickle file of search results")

FLAGS = tf.app.flags.FLAGS

def initialize_model(session, model):
  # ckpt = tf.train.get_checkpoint_state(train_dir)
  # v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
  # if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
      # logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
      # model.saver.restore(session, ckpt.model_checkpoint_path)
  # else:
      # logging.info("Created model with fresh parameters.")
      # session.run(tf.global_variables_initializer())
      # logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
  # return model

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

# Read entire file if num_samples is -1
def load_dataset(tier, num_samples=-1): # tier: 'train', 'dev', 'test'
  premises = []
  premise_lens = []
  hypotheses = []
  hypothesis_lens = []
  goldlabels = []
  with open(pjoin(FLAGS.data_dir, tier + '.ids.premise')) as premise_file, \
       open(pjoin(FLAGS.data_dir, tier + '.ids.hypothesis')) as hypothesis_file, \
       open(pjoin(FLAGS.data_dir, tier + '.goldlabel')) as goldlabel_file:

    # assumes that premise, hypothesis, goldlabel all have same # of lines
    for i, premise_line in enumerate(premise_file):
      if i == num_samples: break

      # line as list of int indices
      premise = map(int, premise_line.strip().split())
      premises.append(premise)
      premise_lens.append(len(premise))
      hypothesis = map(int, hypothesis_file.readline().strip().split())
      hypotheses.append(hypothesis)
      hypothesis_lens.append(len(hypothesis))
      goldlabels.append(convert_to_one_hot(goldlabel_file.readline().strip()))

    return (premises, premise_lens, hypotheses, hypothesis_lens, goldlabels)

def get_save_filename(lr, dropout_keep):
  ntrain_str = str(FLAGS.num_train) if not FLAGS.num_train == -1 else 'all'
  return ('dev' if FLAGS.dev else 'test') + '_numtrain' + ntrain_str + \
                                            '_lr' + str(lr) + \
                                            '_dropoutkeep' + str(dropout_keep)

def run_model(embeddings, train_dataset, eval_dataset, vocab, rev_vocab, lr, dropout_keep, reg_lambda=-1, analyze=False):

  logging.info(FLAGS.__flags)
  logging.info("Learning rate: " + str(lr))
  logging.info("Dropout keep: " + str(dropout_keep))
  logging.info("Reg lambda: " + str(reg_lambda))

  # Reset every time. TODO: we should be using the same graph
  tf.reset_default_graph()
  tf.set_random_seed(1)

  nli = NLISystem(
    pretrained_embeddings = embeddings,
    lr = lr,
    reg_lambda = reg_lambda,
    ff_hidden_size = FLAGS.ff_hidden_size,
    stmt_hidden_size = FLAGS.stmt_hidden_size,
    lstm_hidden_size = FLAGS.lstm_hidden_size,
    num_classes = FLAGS.num_classes,
    ff_num_layers = FLAGS.ff_num_layers,
    tboard_path = FLAGS.tboard_path,
    dropout_keep = dropout_keep,
    bucket = FLAGS.bucket,
    stmt_processor = FLAGS.stmt_processor,
    attentive_matching = FLAGS.attentive_matching,
    max_attentive_matching = FLAGS.max_attentive_matching,
    full_matching = FLAGS.full_matching,
    maxpool_matching = FLAGS.maxpool_matching,
    infer_embeddings = FLAGS.infer_embeddings,
    weight_attention = FLAGS.weight_attention,
    n_bilstm_layers = FLAGS.n_bilstm_layers,
    pool_merge = FLAGS.pool_merge,
    train_embed = FLAGS.train_embed,
    max_grad_norm = FLAGS.max_grad_norm,
    analytic_mode = FLAGS.analysis_path is not None)
  nli.saver = tf.train.Saver() # for saving

  if not os.path.exists(FLAGS.log_dir):
    os.makedirs(FLAGS.log_dir)
    file_handler = logging.FileHandler(pjoin(FLAGS.log_dir, "log.txt"))
    logging.getLogger().addHandler(file_handler)

  with open(os.path.join(FLAGS.log_dir, "flags.json"), 'w') as fout:
    json.dump(FLAGS.__flags, fout)

  # Train and evaluate the model
  with tf.Session() as sess:
    initialize_model(sess, nli)

    # Just get analytic data
    if FLAGS.analysis_path is not None:
      assert FLAGS.restore_path is not None, "Without data to restore, analytics can't be done"
      nli.saver.restore(sess, FLAGS.restore_path)
      analysis = nli.analyze(sess, eval_dataset, rev_vocab, FLAGS.batch_size)
      pickle.dump(analysis, open(FLAGS.analysis_path, "wb"))
      print("Done.")

    # Run and train model
    else:
      if FLAGS.restore_path is not None:
        # nli.saver.restore(sess, pjoin(FLAGS.train_dir, get_save_filename(lr, dropout_keep)))
        nli.saver.restore(sess, FLAGS.restore_path)
        epoch_number, train_accuracy, train_loss = nli.train(sess, train_dataset, rev_vocab, FLAGS.train_dir, FLAGS.batch_size)
      else:
        epoch_number, train_accuracy, train_loss, error = nli.train(sess, train_dataset, rev_vocab, FLAGS.train_dir, FLAGS.batch_size)

        if error:
          nli.saver.save(sess, "train_params/nan_model")
          assert(False)

      # Save the parameters to filej
      # if not FLAGS.validation:
        # nli.saver.save(sess, pjoin(FLAGS.train_dir, get_save_filename(lr, dropout_keep)))
      # else:
        # nli.saver.save(sess, pjoin(FLAGS.validation_dir, get_save_filename(lr, dropout_keep)))

      test_accuracy, avg_test_loss, cm = nli.evaluate_prediction(sess, FLAGS.batch_size, eval_dataset)
      return (epoch_number, train_accuracy, train_loss, test_accuracy, avg_test_loss, cm)

def validate_model(embeddings, train_dataset, eval_dataset, vocab, rev_vocab):
  # Define ranges to randomly sample over
  lr_bounds = [0.00001, 0.001]
  dropout_bounds = [0.5, 1.0]
  num_validation_samples = 10

  results_map = {}
  best_train_accuracy = 0
  best_test_accuracy = 0
  for i in xrange(num_validation_samples):
    lr = np.random.uniform(lr_bounds[0], lr_bounds[1])
    dropout_keep = np.random.uniform(dropout_bounds[0], dropout_bounds[1])

    print("########################################################")
    print("\nRUNNING TRIAL: ", str(i), "\tlr:", lr, "\tdropout:", dropout_keep, "\n")
    idx_tup = (lr, dropout_keep)
    results_map[idx_tup] = run_model(embeddings, train_dataset, eval_dataset, vocab, rev_vocab,
                                     lr, dropout_keep)
    pickle.dump(results_map, open(FLAGS.hyperparameter_grid_search_file, "wb"))
    print("############################")
    print("TRIAL RESULTS: ", "\tlr:", lr, "\tdropout:", dropout_keep)
    print("\tACCURACY: \ttrain:", results_map[idx_tup][1], "\ttest:", results_map[idx_tup][3])
    if results_map[idx_tup][1] > best_train_accuracy:
      best_train_accuracy = results_map[idx_tup][1]
      print("\t\tNew best TRAIN")
    if results_map[idx_tup][3] > best_test_accuracy:
      best_test_accuracy = results_map[idx_tup][3]
      print("\t\tNew best TEST")
    print("########################################################")

def main(_):

  assert(FLAGS.validation or ((FLAGS.dev and not FLAGS.test) or (FLAGS.test and not FLAGS.dev))), "When not validating, must set exaclty one of --dev or --test flag to specify evaluation dataset."
  assert FLAGS.stmt_processor in ["bow", "lstm", "bilstm", "stacked"], "Statement processor must be one of bow, lstm, or bilstm."
  assert not (FLAGS.attentive_matching or FLAGS.max_attentive_matching or FLAGS.full_matching) or FLAGS.stmt_processor in ["lstm", "bilstm", "stacked"], "Statement processor must be lstm or bilstm if attention is used."
  assert not FLAGS.infer_embeddings or (FLAGS.attentive_matching or FLAGS.max_attentive_matching or FLAGS.full_matching), "Attention must be enabled to infer embeddings"

  # SET RANDOM SEED
  np.random.seed(244)

  # Load the two pertinent datasets
  train_dataset = load_dataset('train', FLAGS.num_train)
  if FLAGS.test:
    eval_dataset = load_dataset('test', FLAGS.num_test)
  else:
    eval_dataset = load_dataset('dev', FLAGS.num_dev)

  # Define paths
  embed_path = FLAGS.embed_path or pjoin("data", "snli", "glove.trimmed.{}.npz".format(FLAGS.embedding_size))
  vocab_path = FLAGS.vocab_path or pjoin(FLAGS.data_dir, "vocab.dat")

  # Get vocab and embeddings
  with np.load(embed_path) as embeddings_dict:
    embeddings = embeddings_dict['glove']
    vocab, rev_vocab = initialize_vocab(vocab_path)

    if not FLAGS.validation:
      run_model(embeddings, train_dataset, eval_dataset, vocab, rev_vocab, FLAGS.lr, FLAGS.dropout_keep, FLAGS.reg_lambda)
    else: # purpose = 'validate'
      validate_model(embeddings, train_dataset, eval_dataset, vocab, rev_vocab)

################### ARGUMENT PARSER ##########################################


if __name__ == "__main__":
  tf.app.run()
