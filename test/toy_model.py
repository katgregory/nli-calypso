from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time, logging, shutil, sys, re

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from util import Progbar, minibatches, ConfusionMatrix
from nli_model import Statement, NLISystem

from os.path import join as pjoin

import numpy as np
import logging

def convert_to_one_hot(label):
  if label == '0':
    return np.array([1, 0, 0])
  elif label == '1':
    return np.array([0, 1, 0])
  elif label == '2':
    return np.array([0, 0, 1])
  print ('failed to convert: ' + str(label))
  return 5/0

def load_dataset(tier): # tier: 'test', 'train', 'dev'
  data_dir = "data/toy"
  premises = []
  goldlabels = []
  with open(pjoin(data_dir, tier + '.premise')) as premise_file, \
      open(pjoin(data_dir, tier + '.goldlabel')) as goldlabel_file:
    for line in premise_file:  
      premises.append(line.strip()) 
    for line in goldlabel_file:
      goldlabels.append(convert_to_one_hot(line.strip()))
    return (premises, goldlabels)

class Config:
  ff_hidden_size = 1
  hidden_size = 100
  num_classes = 3
  lr = 0.005
  verbose = True
  LBLS = ['entailment', 'neutral', 'contradiction']
  n_epochs = 1
  logpath = './logs'

  
class ToyModel(object):
  def __init__(self, *args):

    embedding_size, num_classes = args

    # Premise and Hypothesis should be input as matrix of sentence_len x batch_size
    self.premise_placeholder = tf.placeholder(tf.float32, shape=(1, 1), name="Premise-Placeholder")

    # Output labels should be a matrix of batch_size x num_classes
    self.output_placeholder = tf.placeholder(tf.float32, shape=(1, num_classes), name="Output-Placeholder")

    # ==== assemble pieces ====
    with tf.variable_scope("nli"):

      # r1 = tanh(merged W1 + b1)
      with tf.variable_scope("FF-First-Layer"):
        W1 = tf.get_variable("W", shape=(1, Config.ff_hidden_size), initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.Variable(tf.zeros([Config.ff_hidden_size,]), name="b")
        r1 = tf.nn.relu(tf.matmul(self.premise_placeholder, W1) + b1, name="r")

        # tf.summary.histogram("W", W1)
        # tf.summary.histogram("b", b1)
        # tf.summary.histogram("r1", self.preds)

      # # r2 = tanh(r1 W2 + b2)
      with tf.variable_scope("FF-Second-Layer"):
        W2 = tf.get_variable("W", shape=(Config.ff_hidden_size, Config.num_classes), initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.Variable(tf.zeros([Config.num_classes,]), name="b")
        self.preds = tf.nn.relu(tf.matmul(r1, W2) + b2, name="r")

      #   tf.summary.histogram("W", W2)
      #   tf.summary.histogram("b", b2)
      #   tf.summary.histogram("r2", r2)

      # # r3 = tanh(r2 W3 + b3)
      # with tf.variable_scope("FF-Third-Layer"):
      #   W3 = tf.get_variable("W", shape=(Config.ff_hidden_size, Config.num_classes), initializer=tf.contrib.layers.xavier_initializer())
      #   b3 = tf.Variable(tf.zeros([Config.num_classes,]), name="b")
      #   self.preds = tf.nn.relu(tf.matmul(r2, W3) + b3, name="r")

      #   tf.summary.histogram("W", W3)
      #   tf.summary.histogram("b", b3)
      #   tf.summary.histogram("preds", self.preds)

      # prediction before softmax layer
      with tf.variable_scope("FF-Softmax"):

        # for logging purposes only
        self.probs = tf.nn.softmax(self.preds)
        # tf.summary.histogram("probs", probs)

        loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.preds, labels=self.output_placeholder, name="loss")
        self.mean_loss = tf.reduce_mean(loss)

    with tf.name_scope("Optimizer"):
      self.train_op = tf.train.AdamOptimizer(Config.lr).minimize(self.mean_loss)
      # tf.summary.scalar("mean_batch_loss", self.mean_loss)

    # with tf.name_scope("Gradients"):
      # summarize out gradients of loss w.r.t all trainable vars
      # trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
      # trainable_vars = [W1, b1, r1, W2, b2, self.preds, self.probs, self.mean_loss] # manually specify for clarity
      # gradients = tf.gradients(self.mean_loss, trainable_vars)

      # for i, gradient in enumerate(gradients):
        # variable = trainable_vars[i]
        # variable_name = re.sub(r':', "_", variable.name)
        # tf.summary.histogram(variable_name + "/loss_gradients", gradient)
        # tf.summary.scalar(variable_name + "/loss_gradient_norms", tf.sqrt(tf.reduce_sum(tf.square(gradient))))

    print("FINISHED INIT")

  #############################
  # TRAINING
  #############################

  def optimize(self, session, train_premise, train_y):
    premise_arr = np.array([[float(premise)] for premise in train_premise])

    # if hasattr(self, "iteration") and self.iteration % 100 == 0:
    #   premise_stmt = premise_arr[0]
    #   hypothesis_stmt = hypothesis_arr[0]
    #   print("Iteration: ", self.iteration)
    #   print( " ".join([rev_vocab[i] for i in premise_stmt]))
    #   print( " ".join([rev_vocab[i] for i in hypothesis_stmt]))
    #   print(train_y)

    input_feed = {
      self.premise_placeholder: premise_arr.T,
      self.output_placeholder: train_y
    }
    output_feed = [self.probs]
    probs = session.run(output_feed, input_feed)
    print(probs)

    if not hasattr(self, "iteration"): self.iteration = 0
    # self.summary_writer.add_summary(summary, self.iteration)
    self.iteration += 1

  def run_epoch(self, session, dataset):
    # prog = Progbar(target=1 + int(len(dataset[0]) / batch_size))
    for i, batch in enumerate(minibatches(dataset, 1)):
      if Config.verbose and (i % 10 == 0):
        sys.stdout.write(str(i) + "...")
        sys.stdout.flush()
      self.optimize(session, *batch)

  """
  Loop through dataset and call optimize() to train model

  :param session: passed in from train.py
  :param dataset: a representation of data
  :param train_dir: path to the directory where the model checkpoint is saved

  """
  def train(self, session, dataset):
    tic = time.time()
    params = tf.trainable_variables()
    num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
    toc = time.time()
    logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))

    self.summary_op = tf.summary.merge_all()
    self.summary_writer = tf.summary.FileWriter('%s/%s' % (Config.logpath, time.time()), graph=session.graph)
    for epoch in range(Config.n_epochs):
      print("\nEpoch", epoch + 1, "out of", Config.n_epochs)
      self.run_epoch(session, dataset)

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

  def predict(self, session, premises, goldlabel):
    premise_arr = np.array([[float(premise)] for premise in premises])

    input_feed = {
      self.premise_placeholder: premise_arr.T,
      self.output_placeholder: goldlabel
    }

    output_feed = [tf.nn.softmax(self.preds), self.mean_loss]
    output, mean_loss = session.run(output_feed, input_feed)

    if Config.verbose:
      print('predicts:', self.label_to_name(output))
      print('with probabilities: ', output)
      if (np.argmax(goldlabel) != np.argmax(output)):
        print('\t\t\t\t correct:', self.label_to_name(goldlabel))
  
    return np.argmax(goldlabel), np.argmax(output), mean_loss

  def evaluate_prediction(self, session, dataset):
    print("EVALUATING")

    cm = ConfusionMatrix(labels=Config.LBLS)
    total_loss = 0
    total_correct = 0
    for batch in minibatches(dataset, 1):
      gold_idx, predicted_idx, loss = self.predict(session, *batch)
      total_correct += 1 if predicted_idx == gold_idx else 0
      total_loss += loss
      cm.update(gold_idx, predicted_idx)
    print(total_correct / float(len(dataset[0])))
    print(total_loss / float(len(dataset[0])))
    print("Token-level confusion matrix:\n" + cm.as_table())
    print("Token-level scores:\n" + cm.summary())


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

def main(_):
  train_dataset = load_dataset('train')
  test_dataset = load_dataset('test')

  embedding_size = 1
  num_classes = 3

  model = ToyModel(embedding_size, num_classes)
  with tf.Session() as sess:
    initialize_model(sess, model, "")
    print("train")
    model.train(sess, train_dataset)
    print("test")
    model.evaluate_prediction(sess, test_dataset)

if __name__ == "__main__":
  tf.app.run()
