from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time, logging, shutil, sys, re
from nli import NLI
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from util import Progbar, minibatches, ConfusionMatrix

ph = tf.placeholder

# from evaluate import exact_match_score, f1_score

logging.basicConfig(level=logging.INFO)

def get_optimizer(lr, opt="adam"):
  if opt == "adam":
    optfn = tf.train.AdamOptimizer(lr)
  elif opt == "adadelta":
    optfn = tf.train.AdadeltaOptimizer(lr)
  elif opt == "sgd":
    optfn = tf.train.GradientDescentOptimizer(lr)
  else:
    assert (False)
  return optfn

# Given an array of probabilities across the three labels, 
# returns string of the label with the highest probability
# (For debugging purposes only)
def label_to_name(label):
  return {
    '0': "entailment",
    '1': 'neutral',
    '2': 'contradiction'
  }[str(np.argmax(label))]

  
class NLISystem(object):
  def __init__(self, pretrained_embeddings,
               lr,
               reg_lambda,
               ff_hidden_size,
               stmt_hidden_size,
               lstm_hidden_size,
               num_classes,
               dropout_keep,
               tboard_path = None,
               verbose = False):

    # Vars that need to be used globally
    self.tboard_path = tboard_path
    self.verbose = verbose
    self.dropout_keep = dropout_keep
    self.LBLS = ['entailment', 'neutral', 'contradiction']

    # Dimensions
    batch_size = None
    sen_len = None

    # Placeholders
    self.dropout_ph = ph(tf.float32, shape=(), name="Dropout-Placeholder")
    self.premise_ph = ph(tf.int32, shape=(batch_size, sen_len), name="Premise-Placeholder")
    self.hypothesis_ph = ph(tf.int32, shape=(batch_size, sen_len), name="Hypothesis-Placeholder")
    self.output_ph = ph(tf.int32, shape=(batch_size, num_classes), name="Output-Placeholder")    
    embeddings = tf.Variable(pretrained_embeddings, name="Embeddings", dtype=tf.float32)

    # Build neural net
    reg_list = []               # List of variables to regularize    
    premise = NLI.process_stmt(embeddings, self.premise_ph, reg_list, lstm_hidden_size)
    hypothesis = NLI.process_stmt(embeddings, self.hypothesis_ph, reg_list, lstm_hidden_size)
    merged = NLI.merge_processed_stmts(premise, hypothesis, stmt_hidden_size, reg_list)
    preds = NLI.feed_forward(merged, self.dropout_ph, ff_hidden_size, num_classes, reg_list)

    # Loss, optimization
    with tf.variable_scope("FF-Softmax"):        
      self.probs = tf.nn.softmax(preds)
      softmax_loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds,
                                                             labels=self.output_ph, name="loss")
      regularizer = tf.contrib.layers.l2_regularizer(reg_lambda)
      reg_loss = tf.contrib.layers.apply_regularization(regularizer, weights_list=reg_list)
      self.loss = tf.reduce_mean(softmax_loss) + reg_loss

    with tf.name_scope("Optimizer"):
      tf.summary.scalar("mean_batch_loss", self.loss)
      self.train_op = get_optimizer(lr).minimize(self.loss)

  #############################
  # TRAINING
  #############################
  
  def pad_sequences(self, data, max_length):
    ret = []
    for sentence in data:
      new_sentence = sentence[:max_length] + [0] * max(0, (max_length - len(sentence)))
      ret.append(new_sentence)
    return ret

  # premise, hypothesis, label are all lists of ints
  def optimize(self, session, rev_vocab, premise, hypothesis, label):

    if self.verbose and hasattr(self, "iteration") and self.iteration % 100 == 0:
      premise_stmt = premise_arr[0]
      hypothesis_stmt = hypothesis_arr[0]
      print("Iteration: ", self.iteration)
      print( " ".join([rev_vocab[i] for i in premise_stmt]))
      print( " ".join([rev_vocab[i] for i in hypothesis_stmt]))

    premise_max = len(max(premise, key=len))
    hypothesis_max = len(max(hypothesis, key=len))

    premise_arr = np.array(self.pad_sequences(premise, premise_max))
    hypothesis_arr = np.array(self.pad_sequences(hypothesis, hypothesis_max))

    input_feed = {
      self.premise_ph: premise_arr,
      self.hypothesis_ph: hypothesis_arr,
      self.output_ph: label,
      self.dropout_ph: self.dropout_keep
    }

    if self.tboard_path is not None:
      output_feed = [self.summary_op, self.train_op, self.loss, self.probs]

      summary, _, loss, probs = session.run(output_feed, input_feed)
      if not hasattr(self, "iteration"): self.iteration = 0
      self.summary_writer.add_summary(summary, self.iteration)
      self.iteration += 1

    else:
      output_feed = [self.train_op, self.loss, self.probs]
      _, loss, probs = session.run(output_feed, input_feed)

    return loss, probs

  def run_epoch(self, session, dataset, rev_vocab, train_dir, batch_size):
    tic = time.time()
    # prog = Progbar(target=1 + int(len(dataset[0]) / batch_size))
    num_correct = 0
    num_batches = 0
    total_loss = 0
    for i, batch in enumerate(minibatches(dataset, batch_size)):
      if self.verbose and (i % 10 == 0):
        sys.stdout.write(str(i) + "...")
        sys.stdout.flush()
      premises, hypotheses, goldlabels = batch
      loss, probs = self.optimize(session, rev_vocab, premises, hypotheses, goldlabels)
      total_loss += loss
      num_batches += 1

      # Record correctness of training predictions
      correct_predictions = np.equal(np.argmax(probs, axis=1), np.argmax(goldlabels, axis=1))
      num_correct = np.sum(correct_predictions)
      
    toc = time.time()

      # LOGGING CODE
      # if (i * batch_size) % 1000 == 0:
        # print("Training Example: " + str(i * batch_size))
        # print("Loss: " + str(loss))
    train_accuracy = num_correct / float(len(dataset[0]))
    epoch_mean_loss = total_loss / float(num_batches)
    print("Amount of time to run this epoch: " + str(toc - tic) + " secs")
    print("Training accuracy for this epoch: " + str(train_accuracy))
    print("Mean loss for this epoch: " + str(epoch_mean_loss))
    return train_accuracy, epoch_mean_loss


  """
  Loop through dataset and call optimize() to train model

  :param session: passed in from train.py
  :param dataset: a representation of data
  :param train_dir: path to the directory where the model checkpoint is saved

  """
  def train(self, session, dataset, rev_vocab, train_dir, batch_size):
    tic = time.time()
    params = tf.trainable_variables()
    num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
    toc = time.time()
    logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))

    self.summary_op = tf.summary.merge_all()
    if self.tboard_path is not None:
      self.summary_writer = tf.summary.FileWriter('%s/%s' % (self.tboard_path, time.time()), graph=session.graph)
    losses = []
    best_epoch = (-1, 0)
    epoch = 1
    while True:
      print("\nEpoch", epoch)
      curr_accuracy, curr_loss = self.run_epoch(session, dataset, rev_vocab, train_dir, batch_size)
      if curr_accuracy > best_epoch[1]:
        print("\tNEW BEST")
        best_epoch = (epoch, curr_accuracy)
      losses.append(curr_loss)
      epoch += 1

      # TEST FOR CONVERGENCE 
      if len(losses) >= 3 and (max(losses[-3:]) - min(losses[-3:])) <= 0.05: 
        break # TODO: Replace everything with constants

      if epoch > 50: # HARD CUTOFF?
        break

    return (best_epoch[0], best_epoch[1], losses)

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

  def predict(self, session, batch_size, premise, hypothesis, goldlabel):
    premise_max = len(max(premise, key=len))
    hypothesis_max = len(max(hypothesis, key=len))

    premise_arr = np.array(self.pad_sequences(premise, premise_max))
    hypothesis_arr = np.array(self.pad_sequences(hypothesis, hypothesis_max))

    input_feed = {
      self.premise_ph: premise_arr,
      self.hypothesis_ph: hypothesis_arr,
      self.output_ph: goldlabel,
      self.dropout_ph: 1
    }

    output_feed = [self.probs, self.loss]
    probs, loss = session.run(output_feed, input_feed)

    return probs, loss

  # TODO: Actually use the parameter batch_size
  def evaluate_prediction(self, session, batch_size, dataset):
    print("\nEVALUATING")

    cm = ConfusionMatrix(labels=self.LBLS)
    total_loss = 0
    total_correct = 0
    num_batches = 0
    for batch in minibatches(dataset, batch_size):
      probs, loss = self.predict(session, batch_size, *batch)
      _, _, goldlabels = batch
      for i in xrange(len(probs)):
        total_correct += 1 if label_to_name(probs[i]) == label_to_name(goldlabels[i]) else 0

        gold_idx = np.argmax(goldlabels[i])
        predicted_idx = np.argmax(probs[i])
        cm.update(gold_idx, predicted_idx)
      total_loss += loss
      num_batches += 1
    accuracy = total_correct / float(len(dataset[0]))
    print("Accuracy: " + str(accuracy))
    average_loss = total_loss / float(num_batches)
    print("Average Loss: " + str(average_loss))
    print("Token-level confusion matrix:\n" + cm.as_table())
    print("Token-level scores:\n" + cm.summary())
    return (accuracy, average_loss, cm)
