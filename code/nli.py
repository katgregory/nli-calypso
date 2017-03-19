import tensorflow as tf
from functools import partial

xavier = tf.contrib.layers.xavier_initializer

class NLI(object):

  def __init__(self, tblog=False, analytic_mode=False):
    self.reg_list = []
    self.tblog = tblog
    self.analytic_mode = analytic_mode

  """
  Returns bag of words mean of input statement

  :param statement: Statement as list of embeddings of dimensions batch_size x statement_len
   x embedding_size
  will be appended as needed to this list.
  """
  # batch_size x statement_len x embedding_size
  def BOW(self, statement, hidden_size):
    with tf.name_scope("Process_Stmt_BOW"):
      # batch_size x embedding_size
      hidden = tf.reduce_mean(statement, 1)
      if self.tblog: tf.summary.histogram("hidden", hidden)
      return hidden

  """
  Returns LSTM cell for use with NLI.LSTM and NLI.biLSTM methods
  @hidden_size is scalar that specifies hidden size of LSTM cell
  """
  def LSTM_cell(self, hidden_size):
    with tf.name_scope("Process_Stmt_LSTM_cell"):
      return tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
      # return tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=dropout_keep)

  """
  Run inputs through LSTM. Assumes that input statements are padded with zeros. Binds cells with
  closure.

  :param statement: Statement as list of embeddings of dimensions batch_size x statement_len
   x embedding_size
  :param stmt_lens: Length of statements before padding as 1D list with dimension batch_size.
  Dimensions of batch_size x 1.
  :param cell: LSTM cell as returned from NLI.LSTM_cell

  :return: function fn that takes 2 arguments: statement, stmt_len where statement is
  of dimensions batch_size x statement_len x embedding_size and stmt_len is of dimensions
  batch_size x 1.

  fn returns A tuple of (outputs, last_output) where outputs represents all
  LSTM outputs and is of dimensions batch_size x statement_len x hidden_size; and last_output
  represents the last output for each statement in the batch, of dimensions batch_size x hidden_size
  """
  def LSTM(self, lstm_hidden_size):
    cell = self.LSTM_cell(lstm_hidden_size)

    def run(statement, stmt_lens):
      with tf.name_scope("Process_Stmt_LSTM"):
        # dimensions
        batch_size = tf.shape(statement)[0]

        initial_state = cell.zero_state(batch_size, tf.float32)
        # batch_size x statement_len x hidden_size
        rnn_outputs, fin_state = tf.nn.dynamic_rnn(cell, statement,
                                                sequence_length=stmt_lens,
                                                initial_state=initial_state)
        last_rnn_output = tf.gather_nd(rnn_outputs,
                                       tf.pack([tf.range(batch_size), stmt_lens-1], axis=1))
      return rnn_outputs, last_rnn_output

    return run

  """
  Create biLSTM that can run inputs through stacked Bi-LSTM and output concatenation of
  forward and backward final hidden state. Assumes that input statements are padded with zeros.
  Binds cells with closure.

  :param statement: Statement as list of embeddings of dimensions batch_size x statement_len
   x embedding_size
  :param stmt_lens: Length of statements before padding as 1D list with dimension batch_size.
  Dimensions of batch_size x 1.
  :param cell_fw: Forward LSTM cell as returned from NLI.LSTM_cell
  :param cell_bw: Backwards LSTM cell as returned from NLI.LSTM_cell
  :param n_layers: Number of LSTM layers

  :return: function fn that takes 2 arguments: statement, stmt_len where statement is
  of dimensions batch_size x statement_len x embedding_size and stmt_len is of dimensions
  batch_size x 1.

  fn returns a tuple of (outputs, last_output) where outputs represents all biLSTM outputs and is
  of dimensions batch_size x statement_len x (hidden_size * 2); and last_output represents the last
  hidden state for each statement in the batch, of dimensions batch_size x (hidden_size * 2).
  Outputs are concatenations of forward and backward outputs
  """
  def biLSTM(self, lstm_hidden_size, n_layers):
    cell_fw = self.LSTM_cell(lstm_hidden_size)
    cell_bw = self.LSTM_cell(lstm_hidden_size)

    def run(statement, stmt_lens):
      with tf.name_scope("Process_Stmt_Bi-LSTM"):
        # dimensions
        batch_size = tf.shape(statement)[0]

        rnn_inputs = statement
        for layer_i in xrange(n_layers):
          with tf.variable_scope("Process_Stmt_Stacked_Bi-LSTM-Layer%d" % layer_i):
            initial_state_fw = cell_fw.zero_state(batch_size, tf.float32)
            initial_state_bw = cell_bw.zero_state(batch_size, tf.float32)

            rnn_outputs, fin_state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, rnn_inputs,
                                                  sequence_length=stmt_lens,
                                                  initial_state_fw=initial_state_fw,
                                                  initial_state_bw=initial_state_bw)
            rnn_outputs = tf.concat(2, rnn_outputs)
            rnn_inputs = rnn_outputs

        last_rnn_output = tf.gather_nd(rnn_outputs,
                                       tf.pack([tf.range(batch_size), stmt_lens-1], axis=1))
      return rnn_outputs, last_rnn_output

    return run

  """
  Calculates attention matrix for two statements

  :param states1: States of statement 1 as output from an LSTM, biLSTM, etc. Dimensions are
  batch_size x statement1_len x hidden_size
  :param states2: States of statement 2 as output from an LSTM, biLSTM, etc. Dimensions are
  batch_size x statement2_len x hidden_size
  :param weight_attention: If true, add weight in attention calculation

  :return: A matrix of attentions as exponentiated dot products of the hidden states of the
  statements. Dimensions are batch_size x statement1_len x statement2_len
  """
  def attention(self, states1, states2, weight_attention):
    with tf.name_scope("Attention-Matrix"):
      # dimensions
      batch_size = tf.shape(states1)[0]
      statement1_len, hidden_size = states1.get_shape().as_list()[1:3]

      if weight_attention:
        # Reshape to 2D matrices for the first multiplication
        W = tf.get_variable("W", shape=(hidden_size, hidden_size), initializer=xavier())
        statement1_len = tf.shape(states1)[1]

        # states1: batch_size * statement1_len x hidden_size
        # e: batch_size * statement1_len x hidden_size
        states1 = tf.reshape(states1, (batch_size * statement1_len, hidden_size))
        e = tf.matmul(states1, W)

        # Reshape to 3D matrices for the second multiplication
        # states1: batch_size x statement1_len x hidden_size
        # e: batch_size x statement1_len x hidden_size
        states1 = tf.reshape(states1, (batch_size, statement1_len, hidden_size))
        e = tf.reshape(e, (batch_size, statement1_len, hidden_size))

        # e: batch_size x statement1_len x statement2_len
        e = tf.matmul(e, states2, transpose_b=True)
      else:
        # e: batch_size x statement1_len x statement2_len
        e = tf.matmul(states1, states2, transpose_b=True)

      e = tf.clip_by_value(e, clip_value_min=-40, clip_value_max=40) # Fixes NaN error
      e_exp = tf.exp(e)

      return e_exp

  """
  Calculates context vectors for two statements by using weighted similarity.

  :param states1: States of statement 1 as output from an LSTM, biLSTM, etc. Dimensions are
  batch_size x statement1_len x hidden_size
  :param states2: States of statement 2 as output from an LSTM, biLSTM, etc. Dimensions are
  batch_size x statement2_len x hidden_size
  :param e: Attention matrix as returned by nli.attention. Dimensions are batch_size x
  statement1_len x statement2_len

  :return: A tuple of (context1, context2) of context vectors for each of the words in statement 1
  and statement 2 respectively. context1 and context2 have the same dimensions as states1 and
  states2.
  """
  def chen_matching(self, states1, states2, e):
    with tf.name_scope("Chen-Matching"):
      # dimensions
      batch_size = tf.shape(states1)[0]
      statement1_len, hidden_size = states1.get_shape().as_list()[1:3]

      ##############
      # Normalize e
      ##############

      # output of tf.reduce_sum has dimensions batch_size x statement1_len
      # reshape to batch_size x statement1_len x 1 to prepare for broadcast
      magnitude1 = tf.reshape(tf.reduce_sum(e, axis=2), (batch_size, -1, 1))
      # e_norm1: batch_size x statement1_len x statement2_len
      e_norm1 = tf.div(e, magnitude1)
      context1 = tf.matmul(states2, e_norm1)

      # output of tf.reduce_sum has dimensions batch_size x statement2_len
      # reshape to batch_size x 1 x statement2_len to prepare for broadcast
      magnitude2 = tf.reshape(tf.reduce_sum(e, axis=1), (batch_size, 1, -1))
      # e_norm2: batch_size x statement1_len x statement2_len
      e_norm2 = tf.div(e, magnitude2)

      ##############
      # Weighted sums
      ##############
      context1 = tf.matmul(e_norm1, states2)
      context2 = tf.matmul(e_norm2, states1, transpose_a=True)

      return (context1, context2)

  """
  Calculates context vectors for two statements by using max attentive matching.

  :param states1: States of statement 1 as output from an LSTM, biLSTM, etc. Dimensions are
  batch_size x statement1_len x hidden_size
  :param states2: States of statement 2 as output from an LSTM, biLSTM, etc. Dimensions are
  batch_size x statement2_len x hidden_size
  :param e: Attention matrix as returned by nli.attention. Dimensions are batch_size x
  statement1_len x statement2_len

  :return: A tuple of (context1, context2) of context vectors for each of the words in statement 1
  and statement 2 respectively. context1 and context2 have the same dimensions as states1 and
  states2.
  """
  def max_matching(self, states1, states2, e):
    # dimensions
    batch_size = tf.shape(states1)[0]

    # batch_size x statement1_len x 1: reshape for broadcasting
    max1 = tf.reshape(tf.reduce_max(e, axis=2), (batch_size, -1, 1))
    # one hot vectors of batch_size x statement1_len x statement2_len
    indices1 = tf.cast(tf.equal(e, max1), tf.float32)
    # average the best if there are multiple
    indices1 = indices1 / tf.reshape(tf.reduce_sum(indices1, axis=2), (batch_size, -1, 1))
    # batch_size x statement1_len x hidden_size
    context1 = tf.matmul(indices1, states2)

    # batch_size x 1 x statement2_len: reshape for broadcasting
    max2 = tf.reshape(tf.reduce_max(e, axis=1), (batch_size, 1, -1))
    # one hot vectors of batch_size x statement1_len x statement2_len
    indices2 = tf.cast(tf.equal(e, max2), tf.float32)
    # average the best if there are multiple
    indices2 = indices2 / tf.reshape(tf.reduce_sum(indices2, axis=1), (batch_size, 1, -1))
    # batch_size x statement2_len x hidden_size
    context2 = tf.matmul(indices2, states1, transpose_a=True)

    return (context1, context2)

  def multi_perspective(self, W, v1, v2):
    # W: hidden_size x K
    # v1: batch_size x statement1_len x hidden_size
    # v2: batch_size x statement2_len x hidden_size
    batch_size = tf.shape(v1)[0]
    hidden_size = v1.get_shape().as_list()[2]
    K = tf.shape(W)[1]

    # reshape for broadcast
    v1 = tf.reshape(v1, (batch_size, -1, 1, hidden_size, 1))
    v2 = tf.reshape(v2, (batch_size, 1, -1, hidden_size, 1))
    W = tf.reshape(W, (1, 1, 1, hidden_size, K))

    # batch_size x statement1_len x 1 x hidden_size x k
    k1 = tf.mul(v1, W)
    k1 = tf.nn.l2_normalize(k1, 4)

    # batch_size x 1 x statement2_len x hidden_size x k
    k2 = tf.mul(v2, W)
    k2 = tf.nn.l2_normalize(k2, 4)

    # batch_size x statement1_len x statement2_len x hidden_size x k
    r = tf.mul(k1, k2)

    # batch_size x statement1_len x statement2_len x k
    r = tf.reduce_sum(r, axis=3)
    return r

  def full_matching(self, states1, states2, p_last, h_last, K):
    with tf.variable_scope("Full-Matching"):
      batch_size = tf.shape(states1)[0]
      hidden_size = states1.get_shape().as_list()[2]
      W = tf.get_variable('W', shape=(hidden_size, K))

      full_context1 = self.multi_perspective(W, states1, h_last)
      full_context2 = self.multi_perspective(W, states2, p_last)

      # batch_size x statement1_len x k
      context1 = tf.reshape(full_context1, (batch_size, -1, K))
      # batch_size x statement2_len x k
      context2 = tf.reshape(full_context2, (batch_size, -1, K))

      return context1, context2

  """
  Calculates context vectors for two statements by using maxpooling-matching

  :param states1: States of statement 1 as output from an LSTM, biLSTM, etc. Dimensions are
  batch_size x statement1_len x hidden_size
  :param states2: States of statement 2 as output from an LSTM, biLSTM, etc. Dimensions are
  batch_size x statement2_len x hidden_size

  :return: A tuple of (context1, context2) of context vectors for each of the words in statement 1
  and statement 2 respectively. context1 and context2 have the same dimensions as states1 and
  states2.
  """
  def maxpool_matching(self, states1, states2, K):
    with tf.variable_scope("Maxpool-Matching"):
      hidden_size = states1.get_shape().as_list()[2]
      W = tf.get_variable('W', shape=(hidden_size, K))

      # batch_size x statement1_len x statement2_len x k
      context = self.multi_perspective(W, states1, states2)

      # batch_size x statement1_len x k
      context1 = tf.reduce_max(context, axis=2)
      # batch_size x statement2_len x k
      context2 = tf.reduce_max(context, axis=1)

      return context1, context2

  
  """
  Return a new vector that embodies inferred information from context and state vectors
  of a statement. Concatenates the context as needed + runs through FF network.

  :param context: Context vector of statement as returned from NLI.context_tensors. Dimensions
  are batch_size x statement_len x hidden_size
  :param states: States vector of statement as output from an LSTM, biLSTM, etc. Dimensions are
  are batch_size x statement_len x hidden_size
  :param states: Embeddings vector of statement as output from embedding_lookup. Dimensions are
  batch_size x statement_len x embedding_size. Optional.

  :return: A context/state inference vector of dimension batch_size x statement_len x
  hidden_size
  """
  def infer(self, context, states, hidden_size, dropout, embeddings=None):
    with tf.name_scope("Infer"):
      batch_size = tf.shape(context)[0]
      stmt_len = tf.shape(context)[1]

      if embeddings is not None:
        m = tf.concat(2, [context, states, states - context, tf.mul(states, context), embeddings])
      else: m = tf.concat(2, [context, states, states - context, tf.mul(states, context)])

      m_size = m.get_shape().as_list()[2]

      m_reshaped = tf.reshape(m, [batch_size * stmt_len, m_size])
      m_ff = self.feed_forward(m_reshaped, dropout, m_size, hidden_size, 1, tf.nn.relu)
      return tf.reshape(m_ff, [batch_size, stmt_len, hidden_size])

  """
  Calculates Average and Max Pool for each composed vector and concatenates them in preparation
  for classification.

  :param composed: Composed vector of statement1 as returned from an LSTM, biLSTM of inferred
  vector. Dimensions are batch_size x statement_len x hidden_size
  :param composed: Composed vector of statement2 as returned from an LSTM, biLSTM of inferred
  vector. Dimensions are batch_size x statement_len x hidden_size

  :return: A merged vector of dimensions batch_size x (hidden_size * 4)
  """
  def pool_merge(self, composed1, composed2):
    with tf.name_scope("Pool-Merge"):
      avg1 = tf.reduce_mean(composed1, axis=1)
      avg2 = tf.reduce_mean(composed2, axis=1)
      max_pool1 = tf.reduce_max(composed1, axis=1)
      max_pool2 = tf.reduce_max(composed2, axis=1)
      return tf.concat(1, [avg1, max_pool1, avg2, max_pool2])

  """
  Merge two hidden states through concatenation after weighting.

  :param state1: First hidden state to merge
  :param state2: Second hidden state to merge
  :param hidden_size: Output hidden size of each state

  :return: Merged hidden state of dimensions batch_size x (hidden_size * 2)
  """
  def merge_states(self, state1, state2, hidden_size):
    with tf.name_scope("Merge-States"):
      state1_size = state1.get_shape().as_list()[1]
      state2_size = state2.get_shape().as_list()[1]

      # weight hidden layers before merging
      with tf.name_scope("Hidden-Weights"):

        W1 = tf.get_variable("W1", shape=(state1_size, hidden_size), initializer=xavier())
        r1 = tf.matmul(state1, W1)
        if self.tblog: tf.summary.histogram("r1", r1)

        W2 = tf.get_variable("W2", shape=(state2_size, hidden_size), initializer=xavier())
        r2 = tf.matmul(state2, W2)
        if self.tblog: tf.summary.histogram("r2", r2)

      return tf.concat(1, [r1, r2], name="merged")

  """
  Implementation of multi-layer Feed forward network with dropout

  :param input: Initial hidden states to pass through FF network. batch_size x ?
  :param dropout: Dropout keep probability
  :param hidden_size: Hidden size of each layer
  :param output_size: Size of output layer
  :param fn: nonlinearity to use between the layers
  :param num_layers: Number >0 representing number of layers in network

  :return: Output state of dimensions batch_size x output_size
  """
  def feed_forward(self, input, dropout, hidden_size, output_size, num_layers, fn):
    with tf.name_scope("Feed-Forward"):
      input_size = input.get_shape().as_list()[1]
      r = input

      for i in range(num_layers):
        first = (i == 0)
        last = (i == num_layers - 1)

        with tf.variable_scope("FF-Layer-" + str(i)):
          i_size = input_size if first else hidden_size
          o_size = output_size if last else hidden_size
          W = tf.get_variable("W", shape=(i_size, o_size), initializer=xavier())
          b = tf.Variable(tf.zeros([o_size,]), name="b")
          mul = tf.matmul(r, W)
          r = tf.add(mul, b, name="r")

          if not last:
            r = fn(r, name="r-nonlin")
            r = tf.nn.dropout(r, dropout, name="r-dropout")

          if self.tblog: tf.summary.histogram("W", W)
          if self.tblog: tf.summary.histogram("b", b)
          if self.tblog: tf.summary.histogram("r", r)
        self.reg_list.append(W)

      return r

    #    W1 = tf.get_variable("W1", shape=(input_size, hidden_size))
    #    b1 = tf.Variable(tf.zeros([hidden_size,]), name="b1")
    #    mul1 = tf.matmul(r, W1)
    #    r1 = tf.add(mul1, b1, name="r1")
    #    fn(r1, name="r-nonlin")
    #    r1 = tf.nn.dropout(r1, dropout, name="r-dropout")
    #    self.reg_list.append(W1)

    #    W2 = tf.get_variable("W2", shape=(hidden_size, output_size))
    #    b2 = tf.Variable(tf.zeros([output_size,]), name="b2")
    #    mul2 = tf.matmul(r1, W2)
    #    r2 = tf.add(mul2, b2, name="r2")
    #    self.reg_list.append(W2)

    #    return W1, b1, mul1, r1, W2, b2, mul2, r2
