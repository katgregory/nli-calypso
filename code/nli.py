import tensorflow as tf
from functools import partial

xavier = tf.contrib.layers.xavier_initializer

class NLI(object):

  """
  Binds processor function as specified by @processor to a function that takes in 2 arguments:
  a statements and the lengths of its sentences. Creates cells as needed such that every call
  to the returned function will use the same cells.

  :param processor: String, either "lstm", "bilstm", or "bow"
  :param reg_list: List of regularization varibles. Variables that need to be regularized 
  will be appended as needed to this list.

  :return: function that takes 2 arguments: statement, stmt_len where statement is
  of dimensions batch_size x statement_len x embedding_size and stmt_len is of dimensions 
  batch_size x 1
  """
  @staticmethod
  def processor(processor, lstm_hidden_size, reg_list):
    if processor == "lstm":
      lstm_cell = NLI.LSTM_cell(lstm_hidden_size)
      process_stmt = partial(lambda c, d, a, b: NLI.LSTM(a, b, c, d), lstm_cell, reg_list)
    elif processor == "bilstm":
      lstm_cell_fw = NLI.LSTM_cell(lstm_hidden_size)
      lstm_cell_bw = NLI.LSTM_cell(lstm_hidden_size)
      process_stmt = partial(lambda c, d, e, a, b: NLI.biLSTM(a, b, c, d, e),
                             lstm_cell_fw, lstm_cell_bw, reg_list)
    elif processor == "bow": # artificially return (None, hidden_state)
      process_stmt = partial(lambda c, a, b: NLI.BOW(a, b, c), reg_list)
      process_stmt = lambda a, b: (None, process_stmt(a, b))
    return process_stmt



  """
  Returns bag of words mean of input statement

  :param statement: Statement as list of embeddings of dimensions batch_size x statement_len 
   x embedding_size
  :param reg_list: List of regularization varibles. Variables that need to be regularized 
  will be appended as needed to this list.
  """
  # batch_size x statement_len x embedding_size
  @staticmethod
  def BOW(statement, hidden_size, reg_list):
    with tf.name_scope("Process_Stmt_BOW"):
      # batch_size x embedding_size
      hidden = tf.reduce_mean(statement, 1)
      tf.summary.histogram("hidden", hidden)
      return hidden

  """
  Returns LSTM cell for use with NLI.LSTM and NLI.biLSTM methods
  @hidden_size is scalar that specifies hidden size of LSTM cell
  """
  @staticmethod
  def LSTM_cell(hidden_size):
    with tf.name_scope("Process_Stmt_LSTM_cell"):
      return tf.nn.rnn_cell.BasicLSTMCell(hidden_size)

  """
  Run inputs through LSTM. Assumes that input statements are padded with zeros.

  :param statement: Statement as list of embeddings of dimensions batch_size x statement_len 
   x embedding_size
  :param stmt_lens: Length of statements before padding as 1D list with dimension batch_size.
  Dimensions of batch_size x 1.
  :param cell: LSTM cell as returned from NLI.LSTM_cell
  :param reg_list: List of regularization varibles. Variables that need to be regularized 
  will be appended as needed to this list.

  :return: A tuple of (outputs, last_output) where outputs represents all LSTM outputs and is 
  of dimensions batch_size x statement_len x hidden_size; and last_output represents the last
  output for each statement in the batch, of dimensions batch_size x hidden_size
  """
  @staticmethod
  def LSTM(statement, stmt_lens, cell, reg_list):
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

  """
  Run inputs through Bi-LSTM and output concatonation of forward and backward
  final hidden state. Assumes that input statements are padded with zeros.

  :param statement: Statement as list of embeddings of dimensions batch_size x statement_len 
   x embedding_size
  :param stmt_lens: Length of statements before padding as 1D list with dimension batch_size.
  Dimensions of batch_size x 1.
  :param cell_fw: Forward LSTM cell as returned from NLI.LSTM_cell
  :param cell_bw: Backwards LSTM cell as returned from NLI.LSTM_cell
  :param reg_list: List of regularization varibles. Variables that need to be regularized 
  will be appended as needed to this list.

  :return: A tuple of (outputs, last_output) where outputs represents all biLSTM outputs and is 
  of dimensions batch_size x statement_len x (hidden_size * 2); and last_output represents the last
  hidden state for each statement in the batch, of dimensions batch_size x (hidden_size * 2). 
  Outputs are concatenations of forward and backward outputs.
  """
  @staticmethod
  def biLSTM(statement, stmt_lens, cell_fw, cell_bw, reg_list):
    with tf.name_scope("Process_Stmt_Bi-LSTM"):
      # dimensions
      batch_size = tf.shape(statement)[0]

      initial_state_fw = cell_fw.zero_state(batch_size, tf.float32)
      initial_state_bw = cell_bw.zero_state(batch_size, tf.float32)

      rnn_outputs, fin_state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, statement,
                                              sequence_length=stmt_lens,
                                              initial_state_fw=initial_state_fw,
                                              initial_state_bw=initial_state_bw)
      rnn_outputs = tf.concat(2, rnn_outputs)
      last_rnn_output = tf.gather_nd(rnn_outputs,
                                     tf.pack([tf.range(batch_size), stmt_lens-1], axis=1))
    return rnn_outputs, last_rnn_output

  """
  Calculates context vectors for two statements by using weighted similarity.

  :param states1: States of statement 1 as output from an LSTM, biLSTM, etc. Dimensions are
  batch_size x statement1_len x hidden_size
  :param states2: States of statement 2 as output from an LSTM, biLSTM, etc. Dimensions are
  batch_size x statement2_len x hidden_size
  :param weight_attention: If true, add weight in atention calculation

  :return: A tuple of (context1, context2) of context vectors for each of the words in statement 1
  and statement 2 respectively. context1 and context2 have the same dimensions as states1 and 
  states2
  """
  @staticmethod
  def context_tensors(states1, states2, weight_attention):
    # dimensions
    batch_size = tf.shape(states1)[0]
    statement1_len, hidden_size = states1.get_shape().as_list()[1:3]

    # e: batch_size x statement1_len x statement2_len
    if weight_attention: 
      # Reshape to 2D matrices for the first multiplication
      W = tf.get_variable("W", shape=(hidden_size, hidden_size), initializer=xavier())
      statement1_len = tf.shape(states1)[1]
      states1 = tf.reshape(states1, (batch_size * statement1_len, hidden_size))
      e = tf.matmul(states1, W)

      # Reshape to 3D matrices for the second multiplication
      states1 = tf.reshape(states1, (batch_size, statement1_len, hidden_size))
      e = tf.reshape(e, (batch_size, statement1_len, hidden_size))
      e = tf.matmul(e, states2, transpose_b=True)
    else:
      e = tf.matmul(states1, states2, transpose_b=True)
    e_exp = tf.exp(e)

    # output of tf.reduce_sum has dimensions batch_size x statement2_len
    # reshape to batch_size x statement2_len x 1 to prepare for broadcast
    magnitude1 = tf.reshape(tf.reduce_sum(e_exp, axis=1), (batch_size, -1, 1))
    # transpose to batch_size x 1 x statement2_len
    magnitude1 = tf.transpose(magnitude1, perm=[0, 2, 1])
    e_norm1 = tf.div(e_exp, magnitude1)
    context1 = tf.matmul(states2, e_norm1, transpose_a=True, transpose_b=True)
    context1 = tf.transpose(context1, perm=[0, 2, 1])

    # output of tf.reduce_sum has dimensions batch_size x statement1_len
    # reshape to batch_size x statement1_len x 1 to prepare for broadcast
    magnitude2 = tf.reshape(tf.reduce_sum(e_exp, axis=2), (batch_size, -1, 1))
    e_norm2 = tf.div(e_exp, magnitude2)
    context2 = tf.matmul(states1, e_norm2, transpose_a=True)
    context2 = tf.transpose(context2, perm=[0, 2, 1])

    return (context1, context2)

  """
  Return a new vector that embodies inferred information from context and state vectors
  of a statement.

  :param context: Context vector of statement as returned from NLI.context_tensors. Dimensions
  are batch_size x statement_len x hidden_size
  :param states: States vector of statement as output from an LSTM, biLSTM, etc. Dimensions are
  are batch_size x statement_len x hidden_size
  :param states: Embeddings vector of statement as output from embedding_lookup. Dimensions are 
  batch_size x statement_len x embedding_size. Optional.

  :return: A composed context/state inference vector of dimension batch_size x statement_len x 
  (hidden_size * 4 + embedding_size (if included))
  """
  @staticmethod
  def merge_context(context, states, embeddings=None):
    if embeddings is not None:
      return tf.concat(2, [context, states, states - context, tf.mul(states, context), embeddings])
    else: return tf.concat(2, [context, states, states - context, tf.mul(states, context)])

  """
  Merge two hidden states through concatenation after weighting.

  :param state1: First hidden state to merge
  :param state2: Second hidden state to merge
  :param hidden_size: Output hidden size of each state
  :param reg_list: List of regularization varibles. Variables that need to be regularized 
  will be appended as needed to this list.

  :return: Merged hidden state of dimensions batch_size x (hidden_size * 2)
  """
  @staticmethod
  def merge_states(state1, state2, hidden_size, reg_list):
    state1_size = state1.get_shape().as_list()[1]
    state2_size = state2.get_shape().as_list()[1]

    # weight hidden layers before merging
    with tf.variable_scope("Hidden-Weights"):

      W1 = tf.get_variable("W1", shape=(state1_size, hidden_size), initializer=xavier())
      r1 = tf.matmul(state1, W1)
      tf.summary.histogram("r1", r1)

      W2 = tf.get_variable("W2", shape=(state2_size, hidden_size), initializer=xavier())
      r2 = tf.matmul(state2, W2)
      tf.summary.histogram("r2", r2)

    return tf.concat(1, [r1, r2], name="merged")

  """
  Implementation of multi-layer Feed forward network with dropout

  :param input: Initial hidden states to pass through FF network. batch_size x ?
  :param dropout: Dropout keep probability
  :param hidden_size: Hidden size of each layer
  :param output_size: Size of output layer
  :param num_layers: Number >1 representing number of layers in network
  :param reg_list: List of regularization varibles. Variables that need to be regularized 
  will be appended as needed to this list.

  :return: Output state of dimensions batch_size x output_size
  """
  @staticmethod
  def feed_forward(input, dropout, hidden_size, output_size, num_layers, reg_list):
    with tf.variable_scope("Feed-Forward"):
      input_size = input.get_shape().as_list()[1]
      r = input

      for i in range(num_layers):
        i_size = input_size if i == 0 else hidden_size
        o_size = output_size if i == num_layers - 1 else hidden_size
        with tf.variable_scope("FF-Layer-" + str(i)):
          W = tf.get_variable("W", shape=(i_size, o_size), initializer=xavier())
          b = tf.Variable(tf.zeros([o_size,]), name="b")
          r = tf.nn.relu(tf.matmul(r, W) + b, name="r")
          r = tf.nn.dropout(r, dropout)
          tf.summary.histogram("W", W)
          tf.summary.histogram("b", b)
          tf.summary.histogram("r", r)
        reg_list.append(W)

      return r
