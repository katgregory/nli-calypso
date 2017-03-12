import tensorflow as tf

xavier = tf.contrib.layers.xavier_initializer

class NLI(object):
  """
  Run inputs through LSTM and output hidden state.

  :param inputs: Inputs as embeddings tensor

  :return: A hidden state representing the statement

  @inputs is of dimensions batch_size x sentence_size
  @mask is of dimensions batch_size x sentence_size
  return value is of dimensions batch_size x hidden_size
  """
  @staticmethod
  def process_stmt(embeddings, statement, hidden_size, mask, sen_len, reg_list, bow=False):
    with tf.name_scope("Process_Stmt"):
      # batch_size x sentence_size x embedding_size
      embeddings = tf.nn.embedding_lookup(embeddings, statement)

      # If using bag of words, simply average embeddings
      if (bow):
        # batch_size x embedding_size
        hidden = tf.reduce_mean(embeddings, 1)
        tf.summary.histogram("hidden", hidden)
        return hidden

      # If using LSTMs, continue:
      else:
        # temp: 1 for now
        batch_size = 1

        # run LSTM
        cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
        initial_state = cell.zero_state(batch_size, tf.float32)
        # batch_size x sentence_size x hidden_size
        _, states = tf.nn.dynamic_rnn(cell, embeddings, initial_state=initial_state)

        # reshape to batch_size * hidden_size x sentence_size for masking
        states = tf.reshape(tf.transpose(states, perm=[0, 2, 1]), [-1, sen_len])
        return tf.reshape(tf.boolean_mask(states, mask), [-1, hidden_size])

# Scratch work:
  #   lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size, )
  #   stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm] * number_of_layers,
  #     state_is_tuple=False)

  # initial_state = state = stacked_lstm.zero_state(batch_size, tf.float32)
  # for i in range(num_steps):
  #     # The value of state is updated after processing each batch of words.
  #     output, state = stacked_lstm(words[:, i], state)

  #   # The rest of the code.
  #   # ...

  # final_state = state



  @staticmethod
  def merge_processed_stmts(stmt1, stmt2, hidden_size, reg_list):
    stmt1_size = stmt1.get_shape().as_list()[1]
    stmt2_size = stmt2.get_shape().as_list()[1]

    # weight hidden layers before merging
    with tf.variable_scope("Hidden-Weights"):

      W1 = tf.get_variable("W1", shape=(stmt1_size, hidden_size), initializer=xavier())
      r1 = tf.matmul(stmt1, W1)
      tf.summary.histogram("r1", r1)

      W2 = tf.get_variable("W2", shape=(stmt2_size, hidden_size), initializer=xavier())
      r2 = tf.matmul(stmt2, W2)
      tf.summary.histogram("r2", r2)

    return tf.concat(1, [r1, r2], name="merged")

  @staticmethod
  def feed_forward(merged, dropout_ph, hidden_size, num_classes, reg_list):
    with tf.variable_scope("FF"):
      # r1 = tanh(merged W1 + b1)
      with tf.variable_scope("FF-First-Layer"):
        merged_size = merged.get_shape().as_list()[1]
        W1 = tf.get_variable("W", shape=(merged_size, hidden_size), initializer=xavier())
        b1 = tf.Variable(tf.zeros([hidden_size,]), name="b")
        r1 = tf.nn.relu(tf.matmul(merged, W1) + b1, name="r")
        r1_dropout = tf.nn.dropout(r1, dropout_ph)

        tf.summary.histogram("W", W1)
        tf.summary.histogram("b", b1)
        tf.summary.histogram("r1", r1)

      # r2 = tanh(r1 W2 + b2)
      with tf.variable_scope("FF-Second-Layer"):
        W2 = tf.get_variable("W", shape=(hidden_size, hidden_size), initializer=xavier())
        b2 = tf.Variable(tf.zeros([hidden_size,]), name="b")
        r2 = tf.nn.relu(tf.matmul(r1_dropout, W2) + b2, name="r")
        r2_dropout = tf.nn.dropout(r2, dropout_ph)

        tf.summary.histogram("W", W2)
        tf.summary.histogram("b", b2)
        tf.summary.histogram("r2", r2)

      # r3 = tanh(r2 W3 + b3)
      with tf.variable_scope("FF-Third-Layer"):
        W3 = tf.get_variable("W", shape=(hidden_size, num_classes), initializer=xavier())
        b3 = tf.Variable(tf.zeros([num_classes,]), name="b")
        r3 = tf.nn.relu(tf.matmul(r2_dropout, W3) + b3, name="r")
        preds = r3

        tf.summary.histogram("W", W3)
        tf.summary.histogram("b", b3)
        tf.summary.histogram("preds", preds)

      reg_list.extend((W1, W2, W3))
      return preds
