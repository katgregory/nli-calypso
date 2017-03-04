from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from util import minibatches
from util import ConfusionMatrix

# from evaluate import exact_match_score, f1_score

logging.basicConfig(level=logging.INFO)

class Config:
  ff_hidden_size = 200
  num_classes = 3
  lr = 0.01
  verbose = True
  LBLS = ['entailment', 'neutral', 'contradiction']

def get_optimizer(opt="adam"):
  if opt == "adam":
    optfn = tf.train.AdamOptimizer(Config.lr)
  elif opt == "sgd":
    optfn = tf.train.GradientDescentOptimizer(Config.lr)
  else:
    assert (False)
  return optfn

"""
Represent "Premise" or "Hypothesis" LSTM portion of model.
"""
class Statement(object):
  def __init__(self, hidden_size):
    self.cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)

  """
  Run inputs through LSTM and output hidden state.

  :param inputs: Inputs as embeddings

  :return: A hidden state representing the statement

  @inputs is of dimensions sentence_size x batch_size x embedding_size
  return value is of dimensions batch_size x hidden_size
  """
  def process(self, inputs):
    batch_size = tf.shape(inputs)[1]
    initial_state = self.cell.zero_state(batch_size, tf.float32)
    output, state = tf.nn.dynamic_rnn(self.cell, inputs, initial_state=initial_state, time_major=True)

    return state[-1]
  
class NLISystem(object):
  def __init__(self, premise, hypothesis, *args):

    vocab_size, embedding_size, num_classes = args

    # ==== set up placeholder tokens ========

    # Premise and Hypothesis should be input as matrix of sentence_len x batch_size
    self.premise_placeholder = tf.placeholder(tf.int32, shape=(None, None), name="premise")
    self.hypothesis_placeholder = tf.placeholder(tf.int32, shape=(None, None), name="hypothesis")
    self.embeddings_placeholder = tf.placeholder(tf.float32, shape=(vocab_size, embedding_size), name="embeddings")

    # Output labels should be a matrix of batch_size x num_classes
    self.output_placeholder = tf.placeholder(tf.int32, shape=(None, num_classes), name="output")

    # Convert to embeddings; should be matrix of dim sentence_len x batch_size x embedding_size
    premise_embeddings = tf.nn.embedding_lookup(self.embeddings_placeholder, self.premise_placeholder)
    hypothesis_embeddings = tf.nn.embedding_lookup(self.embeddings_placeholder, self.hypothesis_placeholder)

    # Scoping used here violates encapsulation slightly for convenience
    with tf.variable_scope("premise"):
      hp = premise.process(premise_embeddings)
    with tf.variable_scope("hypothesis"):
      hh = hypothesis.process(hypothesis_embeddings)

    # ==== assemble pieces ====
    with tf.variable_scope("nli", initializer=tf.contrib.layers.xavier_initializer()):
      merged = tf.concat(1, [hp, hh])
      
      # r = ReLU(merged W1 + b1)
      merged_size = merged.get_shape().as_list()[1]
      W1 = tf.get_variable("W1", shape=(merged_size, Config.ff_hidden_size))
      b1 = tf.get_variable("b1", shape=(Config.ff_hidden_size,))
      r = tf.nn.relu(tf.matmul(merged, W1) + b1)
      
      # softmax(rW2 + b2)
      W2 = tf.get_variable("W2", shape=(Config.ff_hidden_size, Config.num_classes))
      b2 = tf.get_variable("b2", shape=(Config.num_classes,))

      # prediction before softmax layer
      self.preds = tf.matmul(r, W2) + b2

      self.add_train_op()

  def add_train_op(self):
    self.loss = tf.nn.softmax_cross_entropy_with_logits(self.preds, self.output_placeholder)
    self.train_op = get_optimizer().minimize(self.loss)
    
  #############################
  # TRAINING
  #############################

  def pad_sequences(self, data, max_length):
    ret = []
    for sentence in data:
      new_sentence = sentence[:max_length] + [0] * max(0, (max_length - len(sentence)))
      ret.append(new_sentence)
    return ret


  def optimize(self, session, embeddings, train_premise, train_hypothesis, train_y):
    premise_arr = [[int(word_idx) for word_idx in premise.split()] for premise in train_premise]
    hypothesis_arr = [[int(word_idx) for word_idx in hypothesis.split()] for hypothesis in train_hypothesis]
    
    premise_max = len(max(train_premise, key=len).split())
    hypothesis_max = len(max(train_premise, key=len).split())

    premise_arr = np.array(self.pad_sequences(premise_arr, premise_max))
    hypothesis_arr = np.array(self.pad_sequences(hypothesis_arr, hypothesis_max))


    input_feed = {
      self.premise_placeholder: premise_arr.T,
      self.hypothesis_placeholder: hypothesis_arr.T,
      self.embeddings_placeholder: embeddings,
      self.output_placeholder: train_y
    }

    output_feed = [self.train_op]
    outputs = session.run(output_feed, input_feed)

    return outputs

  """
  Loop through dataset and call optimize() to train model

  :param session: passed in from train.py
  :param dataset: a representation of data
  :param train_dir: path to the directory where the model checkpoint is saved

  """
  def train(self, session, dataset, train_dir, embeddings, batch_size):
    tic = time.time()
    params = tf.trainable_variables()
    num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
    toc = time.time()
    logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))

    for i, batch in enumerate(minibatches(dataset, batch_size)):
      if Config.verbose and (i % 10 == 0):
        print("Training batch", i)
      self.optimize(session, embeddings, *batch)

  #############################
  # VALIDATION
  #############################

  def test(self, session, valid_x, valid_y):
    input_feed = {}

    # fill in this feed_dictionary like:
    # input_feed['valid_x'] = valid_x

    output_feed = [self.preds]

    outputs = session.run(output_feed, input_feed)

    return outputs

  def validate(self, sess, valid_dataset):
    """
      Iterate through the validation dataset and determine what
      the validation cost is.

      This method calls self.test() which explicitly calculates validation cost.

      How you implement this function is dependent on how you design
      your data iteration function

      :return:
      """
    valid_cost = 0

    for valid_x, valid_y in valid_dataset:
      valid_cost = self.test(sess, valid_x, valid_y)

    return valid_cost

  #############################
  # TEST
  #############################

  # Given an array of probabilities across the three labels, 
  # returns string of the label with the highest probability
  # (For debugging purposes only)
  def label_to_name(self, label):
    return {
      '0': "entailment",
      '1': 'neutral',
      '2': 'contradiction'
    }[str(np.argmax(label))]

  def predict(self, session, embeddings, premise, hypothesis, goldlabel):
    input_feed = {
      self.premise_placeholder: np.array([[int(x) for x in premise[0].split()]]).T,
      self.hypothesis_placeholder: np.array([[int(x) for x in hypothesis[0].split()]]).T,
      self.output_placeholder: goldlabel,
      self.embeddings_placeholder: embeddings
    }

    output_feed = [tf.nn.softmax(self.preds), self.loss]
    output, loss = session.run(output_feed, input_feed)

    if Config.verbose:
      print('predicts:', self.label_to_name(output))
      if (np.argmax(goldlabel) != np.argmax(output)):
        print('\t\t\tcorrect:', self.label_to_name(goldlabel))

    return np.argmax(goldlabel), np.argmax(output), loss

  def evaluate_prediction(self, session, dataset, embeddings):
    print("EVALUATING")

    cm = ConfusionMatrix(labels=Config.LBLS)
    total_loss = 0
    total_correct = 0
    for batch in minibatches(dataset, 1):
      gold_idx, predicted_idx, loss = self.predict(session, embeddings, *batch)
      total_correct += 1 if predicted_idx == gold_idx else 0
      total_loss += loss
      cm.update(gold_idx, predicted_idx)
    print(total_correct / float(len(dataset[0])))
    print(total_loss / float(len(dataset[0])))
    print("Token-level confusion matrix:\n" + cm.as_table())
    print("Token-level scores:\n" + cm.summary())
