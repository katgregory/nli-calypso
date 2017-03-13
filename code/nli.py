import tensorflow as tf

xavier = tf.contrib.layers.xavier_initializer

class NLI(object):
  """
  Returns bag of words mean of input statement

  :param statement: Statement as list of embeddings of dimensions batch_size x sentence_size 
   x embedding_size
  :param reg_list: List of regularization varibles. Variables that need to be regularized 
  will be appended as needed to this list.
  """
  # batch_size x sentence_size x embedding_size
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

  :param statement: Statement as list of embeddings of dimensions batch_size x sentence_size 
   x embedding_size
  :param stmt_lens: Length of statements before padding as 1D list with dimension batch_size
  :param cell: LSTM cell as returned from NLI.LSTM_cell
  :param reg_list: List of regularization varibles. Variables that need to be regularized 
  will be appended as needed to this list.

  :return: A hidden state representing the statement of dimensions batch_size x hidden_size where
  hidden_size is hidden size of @cell
  """
  @staticmethod
  def LSTM(statement, stmt_lens, cell, reg_list):
    with tf.name_scope("Process_Stmt_LSTM"):
      # dimensions
      batch_size = tf.shape(statement)[0]
      sen_size = tf.shape(statement)[1]

      initial_state = cell.zero_state(batch_size, tf.float32)

      # batch_size x sentence_size x hidden_size
      rnn_outputs, fin_state = tf.nn.dynamic_rnn(cell, statement,
                                              sequence_length=stmt_lens,
                                              initial_state=initial_state)
      last_rnn_output = tf.gather_nd(rnn_outputs,
                                     tf.pack([tf.range(batch_size), stmt_lens-1], axis=1))
    return last_rnn_output

  """
  Run inputs through Bi-LSTM and output concatonation of forward and backward
  final hidden state. Assumes that input statements are padded with zeros.

  :param statement: Statement as list of embeddings of dimensions batch_size x sentence_size 
   x embedding_size
  :param stmt_lens: Length of statements before padding as 1D list with dimension batch_size
  :param cell_fw: Forward LSTM cell as returned from NLI.LSTM_cell
  :param cell_bw: Backwards LSTM cell as returned from NLI.LSTM_cell
  :param reg_list: List of regularization varibles. Variables that need to be regularized 
  will be appended as needed to this list.

  :return: A hidden state representing the statement. Concatenation of final 
  forward and backward hidden states. Dimensions are batch_size x (hidden_size * 2)
  """
  @staticmethod
  def biLSTM(statement, stmt_lens, cell_fw, cell_bw, reg_list):
    with tf.name_scope("Process_Stmt_Bi-LSTM"):
      # dimensions
      batch_size = tf.shape(statement)[0]
      sen_size = tf.shape(statement)[1]

      initial_state_fw = cell_fw.zero_state(batch_size, tf.float32)
      initial_state_bw = cell_bw.zero_state(batch_size, tf.float32)

      rnn_outputs, fin_state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, statement,
                                              sequence_length=stmt_lens,
                                              initial_state_fw=initial_state_fw,
                                              initial_state_bw=initial_state_bw)
      rnn_outputs = tf.concat(2, rnn_outputs)
      last_rnn_output = tf.gather_nd(rnn_outputs,
                                     tf.pack([tf.range(batch_size), stmt_lens-1], axis=1))
    return last_rnn_output

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
  :param reg_list: List of regularization varibles. Variables that need to be regularized 
  will be appended as needed to this list.

  :return: Output state of dimensions batch_size x output_size
  """
  @staticmethod
  def feed_forward(input, dropout, hidden_size, output_size, reg_list):
    with tf.variable_scope("FF"):
      # r1 = tanh(input W1 + b1)
      with tf.variable_scope("FF-First-Layer"):
        input_size = input.get_shape().as_list()[1]
        W1 = tf.get_variable("W", shape=(input_size, hidden_size), initializer=xavier())
        b1 = tf.Variable(tf.zeros([hidden_size,]), name="b")
        r1 = tf.nn.relu(tf.matmul(input, W1) + b1, name="r")
        r1_dropout = tf.nn.dropout(r1, dropout)

        tf.summary.histogram("W", W1)
        tf.summary.histogram("b", b1)
        tf.summary.histogram("r1", r1)

      # r2 = tanh(r1 W2 + b2)
      with tf.variable_scope("FF-Second-Layer"):
        W2 = tf.get_variable("W", shape=(hidden_size, hidden_size), initializer=xavier())
        b2 = tf.Variable(tf.zeros([hidden_size,]), name="b")
        r2 = tf.nn.relu(tf.matmul(r1_dropout, W2) + b2, name="r")
        r2_dropout = tf.nn.dropout(r2, dropout)

        tf.summary.histogram("W", W2)
        tf.summary.histogram("b", b2)
        tf.summary.histogram("r2", r2)

      # r3 = tanh(r2 W3 + b3)
      with tf.variable_scope("FF-Third-Layer"):
        W3 = tf.get_variable("W", shape=(hidden_size, output_size), initializer=xavier())
        b3 = tf.Variable(tf.zeros([output_size,]), name="b")
        r3 = tf.nn.relu(tf.matmul(r2_dropout, W3) + b3, name="r")
        preds = r3

        tf.summary.histogram("W", W3)
        tf.summary.histogram("b", b3)
        tf.summary.histogram("preds", preds)

      reg_list.extend((W1, W2, W3))
      return preds
