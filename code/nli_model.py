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
from tqdm import *
import cPickle as pickle

ph = tf.placeholder

# from evaluate import exact_match_score, f1_score

logging.basicConfig(level=logging.INFO)

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
               ff_num_layers,
               dropout_keep,
               bucket,
               stmt_processor,
               attentive_matching,
               infer_embeddings,
               weight_attention,
               n_bilstm_layers,
               train_embed,
               pool_merge,
               max_grad_norm,
               analytic_mode = False,
               tboard_path = None,
               verbose = False):

    # Vars that need to be used globally
    self.tboard_path = tboard_path
    self.verbose = verbose
    self.dropout_keep = dropout_keep
    self.LBLS = ['entailment', 'neutral', 'contradiction']
    self.bucket = bucket
    self.analytic_mode = analytic_mode

    # Dimensions
    batch_size = None
    sen_len = None

    # Placeholders
    self.dropout_ph = ph(tf.float32, shape=(), name="Dropout-Placeholder")
    self.premise_ph = ph(tf.int32, shape=(batch_size, sen_len), name="Premise-Placeholder")
    self.premise_len_ph = ph(tf.int32, shape=(batch_size,), name="Premise-Len-Placeholder")
    self.hypothesis_ph = ph(tf.int32, shape=(batch_size, sen_len), name="Hypothesis-Placeholder")
    self.hypothesis_len_ph = ph(tf.int32, shape=(batch_size,), name="Hypothesis-Len-Placeholder")
    self.output_ph = ph(tf.int32, shape=(batch_size, num_classes), name="Output-Placeholder")

    embed_fn = tf.Variable if train_embed else tf.constant
    embeddings = embed_fn(pretrained_embeddings, name="Embeddings", dtype=tf.float32)

    ##########################
    # Build neural net
    ##########################
    nli = NLI(tblog=True, analytic_mode=analytic_mode)

    ####################
    # Embedding lookup
    ####################
    premise_embed = tf.nn.embedding_lookup(embeddings, self.premise_ph)
    hypothesis_embed = tf.nn.embedding_lookup(embeddings, self.hypothesis_ph)

    ####################
    # Process statements
    ####################

    # Configure LSTM and process_stmt functions based on flags
    with tf.variable_scope("Process") as scope:
      if stmt_processor == "lstm":
        process_stmt = nli.LSTM(lstm_hidden_size)
      elif stmt_processor == "bilstm":
        process_stmt = nli.biLSTM(lstm_hidden_size, n_bilstm_layers)
      elif stmt_processor == "bow":
        process_stmt = lambda a, b: (None, nli.BOW(a, b))
      else: assert False, "Statement processor invalid"

      p_states, p_last = process_stmt(premise_embed, self.premise_len_ph)
      scope.reuse_variables()
      h_states, h_last = process_stmt(hypothesis_embed, self.hypothesis_len_ph)

    ####################
    # Attention
    ####################
    if attentive_matching:
      with tf.name_scope("Attention"):
        # Context generation
        with tf.variable_scope("Context") as scope:
          if nli.analytic_mode:
            self.e, ret = nli.context_tensors(p_states, h_states, weight_attention)
          else: ret = nli.context_tensors(p_states, h_states, weight_attention)
          p_context, h_context = ret

        # Inference
        with tf.variable_scope("Inference") as scope:
          p_inferred = nli.infer(p_context, p_states, lstm_hidden_size, self.dropout_ph,
                                 premise_embed if infer_embeddings else None)
          scope.reuse_variables()
          h_inferred = nli.infer(h_context, h_states, lstm_hidden_size, self.dropout_ph,
                                 hypothesis_embed if infer_embeddings else None)

        # Composition
        with tf.variable_scope("Composition") as scope:
          if stmt_processor == "lstm":
            compose = nli.LSTM(lstm_hidden_size)
          elif stmt_processor == "bilstm":
            compose = nli.biLSTM(lstm_hidden_size, n_bilstm_layers)

          p_composed, p_last = compose(p_inferred, self.premise_len_ph)
          scope.reuse_variables()
          h_composed, h_last = compose(h_inferred, self.hypothesis_len_ph)

    ####################
    # Merge
    ####################
    if pool_merge and attentive_matching: merged = nli.pool_merge(p_composed, h_composed)
    else: merged = nli.merge_states(p_last, h_last, stmt_hidden_size)

    ####################
    # Loss
    ####################
    with tf.variable_scope("FF-Softmax"):
      # Feed-Forward
      preds = nli.feed_forward(merged, self.dropout_ph, ff_hidden_size, num_classes,
                               ff_num_layers, tf.nn.tanh)

      # Softmax
      self.probs = tf.nn.softmax(preds)
      softmax_loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds,
                                                             labels=self.output_ph, name="loss")
      self.loss = tf.reduce_mean(softmax_loss)

      tf.summary.histogram("preds", preds)
      tf.summary.histogram("probs", self.probs)

      # Regularization
      if reg_lambda >= 0:
        regularizer = tf.contrib.layers.l2_regularizer(reg_lambda)
        reg_loss = tf.contrib.layers.apply_regularization(regularizer, weights_list=nli.reg_list)
        self.loss += reg_loss

    ####################
    # Optimizer
    ####################
    with tf.name_scope("Optimizer"):
      tf.summary.scalar("mean_batch_loss", self.loss)

      # Gradient clipping
      optimizer = tf.train.AdamOptimizer(lr)
      grads_and_vars = optimizer.compute_gradients(self.loss)
      self.gradients = [x[0] for x in grads_and_vars]

      if (max_grad_norm >= 0):
          self.gradients, _ = tf.clip_by_global_norm(self.gradients, max_grad_norm)
      self.train_op = optimizer.apply_gradients([(self.gradients[i], grads_and_vars[i][1]) for i in xrange(len(grads_and_vars))])

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
  def optimize(self, session, rev_vocab, premise, premise_len, hypothesis, hypothesis_len, label):

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
      self.premise_len_ph: premise_len,
      self.hypothesis_ph: hypothesis_arr,
      self.hypothesis_len_ph: hypothesis_len,
      self.output_ph: label,
      self.dropout_ph: self.dropout_keep
    }

    if self.tboard_path is not None:
      output_feed = [self.summary_op, self.train_op, self.loss, self.probs]
      summary, _, loss, probs = session.run(output_feed, input_feed)
      self.summary_writer.add_summary(summary, self.iteration)

    else:
      output_feed = [self.train_op, self.loss, self.probs]
      _, loss, probs = session.run(output_feed, input_feed)

    # if loss != loss: # Nan - aka we f-ed up.
      # print('\nBATCH LOSS IS NAN!! Printing out...')

      # f = open("vars", 'w')
      # allVars = [e_exp, mag1, e_norm1, mag2, e_norm2]
      # names = ["e_exp", "mag1", "e_norm1", "mag2", "e_norm2"]
      # for i, varp in enumerate(allVars):
      #   print("NAN")
      #   print(names[i] + str(np.argwhere(np.isnan(varp))))
      #   print("INF")
      #   print(names[i] + str(np.argwhere(np.isinf(varp))))
      #   pickle.dump(varp, f)
      # f.close()

      # print('Loss:', loss, '\n')
      # for i,x in enumerate(probs):
      #   if x[0] != x[0] or x[1] != x[1] or x[2] != x[2]:
      #     print('\n\tCulprit:')
      #     print('\t\tPremise:', premises[i])
      #     print('\t\tPremiseLen:', premise_lens[i])
      #     print('\t\tHypothesis:', hypotheses[i])
      #     print('\t\tHypothesisLen:', hypothesis_lens[i])
      # print('correct_predictions', correct_predictions, '\n')
      # print('gradients:', gradients, '\n')
      # return -1, -1, True


    return loss, probs, False

  def run_epoch(self, session, dataset, rev_vocab, train_dir, batch_size):
    tic = time.time()
    # prog = Progbar(target=1 + int(len(dataset[0]) / batch_size))
    num_correct = 0
    num_batches = 0
    total_loss = 0

    with tqdm(total=int(len(dataset[0]))) as pbar:
      for i, batch in enumerate(minibatches(dataset, batch_size, bucket=self.bucket)):
        self.iteration += batch_size # for tensorboard
        if self.verbose and (i % 10 == 0):
          sys.stdout.write(str(i) + "...")
          sys.stdout.flush()
        premises, premise_lens, hypotheses, hypothesis_lens, goldlabels = batch
        loss, probs, error = self.optimize(session, rev_vocab, premises, premise_lens, hypotheses, hypothesis_lens, goldlabels)
        total_loss += loss
        num_batches += 1

        # Record correctness of training predictions
        correct_predictions = np.equal(np.argmax(probs, axis=1), np.argmax(goldlabels, axis=1))
        num_correct += np.sum(correct_predictions)
        pbar.update(batch_size)

    toc = time.time()

      # LOGGING CODE
      # if (i * batch_size) % 1000 == 0:
        # print("Training Example: " + str(i * batch_size))
        # print("Loss: " + str(loss))
    train_accuracy = num_correct / float(len(dataset[0]))
    epoch_mean_loss = total_loss / float(num_batches)

    if epoch_mean_loss != epoch_mean_loss: # Nan - aka we f-ed up.
      print('\nMEAN LOSS IS NAN!! Printing out...')
      print('Mean Loss:', epoch_mean_loss, '\n')
      return -1, -1, True

    print("Amount of time to run this epoch: " + str(toc - tic) + " secs")
    print("Training accuracy for this epoch: " + str(train_accuracy))
    print("Mean loss for this epoch: " + str(epoch_mean_loss))
    return train_accuracy, epoch_mean_loss, False


  def analyze(self, session, dataset, rev_vocab, batch_size):
    assert self.analytic_mode, "Analytic mode must be enabled to call analyze"
    with tqdm(total=int(len(dataset[0]))) as pbar:
      premise_analysis = []
      hypothesis_analysis = []
      e_analysis = []
      for i, batch in enumerate(minibatches(dataset, batch_size, bucket=self.bucket)):
        premises, premise_lens, hypotheses, hypothesis_lens, goldlabels = batch

        premise_max = len(max(premises, key=len))
        hypothesis_max = len(max(hypotheses, key=len))
        premise_arr = np.array(self.pad_sequences(premises, premise_max))
        hypothesis_arr = np.array(self.pad_sequences(hypotheses, hypothesis_max))

        input_feed = {
          self.premise_ph: premise_arr,
          self.premise_len_ph: premise_lens,
          self.hypothesis_ph: hypothesis_arr,
          self.hypothesis_len_ph: hypothesis_lens,
          self.output_ph: goldlabels,
          self.dropout_ph: self.dropout_keep
        }

        output_feed = [self.loss, self.probs, self.e]
        loss, probs, e = session.run(output_feed, input_feed)

        premise_analysis.append([[rev_vocab[i] for i in premise] for premise in premises])
        hypothesis_analysis.append([[rev_vocab[i] for i in hypothesis] for hypothesis in hypotheses])
        e_analysis.append(e)

        pbar.update(batch_size)

    return (premise_analysis, hypothesis_analysis, e_analysis)

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

    self.iteration = 0
    self.summary_op = tf.summary.merge_all()
    if self.tboard_path is not None:
      self.summary_writer = tf.summary.FileWriter('%s/%s' % (self.tboard_path, time.time()), graph=session.graph)

    losses = []
    best_epoch = (-1, 0)
    epoch = 1
    while True:
      print("\nEpoch", epoch)
      curr_accuracy, curr_loss, error = self.run_epoch(session, dataset, rev_vocab, train_dir, batch_size)
      if error:
        return (-1, -1, -1, True)
      if curr_accuracy > best_epoch[1]:
        print("\tNEW BEST")
        best_epoch = (epoch, curr_accuracy)
      losses.append(curr_loss)
      epoch += 1

      if curr_loss != curr_loss: # Nan - aka we f-ed up.
        print('\nBATCH LOSS IS NAN!! Printing out...')
        print('Loss:', curr_loss, '\n')
        return -1, -1, -1, True
      self.saver.save(session, 'train_params/epoch_model' + str(epoch)) # Only save parameters if we don't crash

      # TEST FOR CONVERGENCE
      if len(losses) >= 10 and (max(losses[-3:]) - min(losses[-3:])) <= 0.03:
        break 

      if epoch > 50: # HARD CUTOFF?
        break

    return (best_epoch[0], best_epoch[1], losses, False)

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

  def predict(self, session, batch_size, batch):
    premise, premise_len, hypothesis, hypothesis_len, goldlabel = batch
    premise_max = len(max(premise, key=len))
    hypothesis_max = len(max(hypothesis, key=len))

    premise_arr = np.array(self.pad_sequences(premise, premise_max))
    hypothesis_arr = np.array(self.pad_sequences(hypothesis, hypothesis_max))

    input_feed = {
      self.premise_ph: premise_arr,
      self.premise_len_ph: premise_len,
      self.hypothesis_ph: hypothesis_arr,
      self.hypothesis_len_ph: hypothesis_len,
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
    for batch in minibatches(dataset, batch_size, bucket=self.bucket):
      probs, loss = self.predict(session, batch_size, batch)
      _, _, _, _, goldlabels = batch
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
