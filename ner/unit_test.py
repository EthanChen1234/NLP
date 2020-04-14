import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.crf import crf_log_likelihood

# case1, test model
char_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None])  # [batch_size, num_steps], [3, 2]
seg_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None])
targets_input = tf.placeholder(dtype=tf.int32, shape=[None, None])
dropout = tf.placeholder(dtype=tf.float32, name="Dropout")

NUM_CHARS = 4680
CHARS_DIM = 5  # 100
NUM_SEG = 4
SEG_DIM = 3  # 20
LSTM_DIM = 10  # 100
NUM_TAGS = 15
BATCH_SIZE = tf.shape(char_inputs)[0]
NUM_STEPS = tf.shape(char_inputs)[-1]  # 句子长度

# embedding layer  # [batch_size, num_steps, emb_size], emd_size = chars_dim + seg_dim
embedding = []
char_lookup = tf.get_variable(name="char_embedding",
                              shape=[NUM_CHARS, CHARS_DIM],
                              initializer=initializers.xavier_initializer())  # shape [4680, 100]
embedding.append(tf.nn.embedding_lookup(char_lookup, char_inputs))

seg_lookup = tf.get_variable(name="seg_embedding",
                             shape=[NUM_SEG, SEG_DIM],
                             initializer=initializers.xavier_initializer())  # shape [4, 20]
embedding.append(tf.nn.embedding_lookup(seg_lookup, seg_inputs))
embed = tf.concat(embedding, axis=-1)


# Bilstm Layer  # [batch_size, num_steps, 2*lstm_dim]
used = tf.sign(tf.abs(char_inputs))
length = tf.reduce_sum(used, reduction_indices=1)
lengths = tf.cast(length, tf.int32)

model_inputs = tf.nn.dropout(embed, rate=1-dropout)
lstm_cell = {}
for direction in ["forward", "backward"]:
    with tf.variable_scope(direction):
        lstm_cell[direction] = tf.contrib.rnn.CoupledInputForgetGateLSTMCell(10,
                                                                             use_peepholes=True,
                                                                             initializer=initializers.xavier_initializer(),
                                                                             state_is_tuple=True)
outputs, output_states = tf.nn.bidirectional_dynamic_rnn(lstm_cell["forward"],
                                                         lstm_cell["backward"],
                                                         model_inputs,
                                                         dtype=tf.float32,
                                                         sequence_length=lengths)
bilstm_outputs = tf.concat(outputs, axis=2)  # 3 x 2 x 20, batch_size x num_steps x (lstm_units x 2)


# projection
# hidden, [batch_size x num_steps, lstm_dim], [2x3, 10]
with tf.variable_scope('hidden'):
    h_w = tf.get_variable('h_w', shape=[LSTM_DIM*2, LSTM_DIM], dtype=tf.float32,
                          initializer=initializers.xavier_initializer())
    h_b = tf.get_variable('h_b', shape=[LSTM_DIM], dtype=tf.float32, initializer=initializers.xavier_initializer())
    bilstm_reshaped = tf.reshape(bilstm_outputs, shape=[-1, LSTM_DIM*2])  # -1 = batch_size x num_steps
    hidden = tf.tanh(tf.matmul(bilstm_reshaped, h_w) + h_b)

# predict of tags, [batch_size, num_steps, num_tags]
with tf.variable_scope('logits'):
    l_w = tf.get_variable('l_w', shape=[LSTM_DIM, NUM_TAGS], dtype=tf.float32,
                          initializer=initializers.xavier_initializer())
    l_b = tf.get_variable('l_b', shape=[NUM_TAGS], dtype=tf.float32, initializer=initializers.xavier_initializer())
    predict_raw = tf.matmul(hidden, l_w) + l_b  # [ batch_size*num_steps, num_tags]
    predict = tf.reshape(predict_raw, [-1, NUM_STEPS, NUM_TAGS])


# loss layer
with tf.variable_scope('crf_loss'):
    small = -1000.0
    # pad logits for crf loss
    start_logits = tf.concat([small*tf.ones(shape=[BATCH_SIZE, 1, NUM_TAGS]),
                              tf.zeros(shape=[BATCH_SIZE, 1, 1])], axis=-1)  # [batch_size, 1, num_tags+1]
    pad_logits = tf.cast(small*tf.ones([BATCH_SIZE, NUM_STEPS, 1]), tf.float32)  # [batch_size, num_steps, 1]
    logits = tf.concat([predict, pad_logits], axis=-1)  # [batch_size, num_steps, num_tags+1]
    logits = tf.concat([start_logits, logits], axis=1)  # [batch_size, num_steps+1, num_tags+1]

    temp = tf.cast(NUM_TAGS * tf.ones([BATCH_SIZE, 1]), tf.int32)  # [[15], [15], [15]]
    targets = tf.concat([temp, targets_input], axis=-1)  # [batch_size, num_steps+1]

    trans = tf.get_variable('transactions', shape=[NUM_TAGS+1, NUM_TAGS+1],
                            initializer=initializers.xavier_initializer())  # [16, 16]
    log_likelihood, tans_out = crf_log_likelihood(inputs=logits,
                                                  tag_indices=targets,
                                                  transition_params=trans,
                                                  sequence_lengths=lengths+1)
    loss = tf.reduce_mean(-log_likelihood)
    # https://github.com/EthanChen1234/docs/blob/r1.13/site/en/api_docs/python/tf/contrib/crf/crf_log_likelihood.md

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    # embedding layer
    print(sess.run(loss, feed_dict={char_inputs: [[23, 24], [2, 1], [30, 41]],
                                    seg_inputs: [[1, 3], [1, 3], [1, 3]],
                                    targets_input: [[0, 0], [0, 0], [2, 3]],
                                    dropout: 0.5}))
