import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers

# case1, test model
# embedding layer  # [batch_size, num_steps, emb_size]
embedding = []
char_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None])
seg_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None])
char_lookup = tf.get_variable(name="char_embedding",
                              shape=[4680, 5],
                              initializer=initializers.xavier_initializer())  # shape [4680, 100]
embedding.append(tf.nn.embedding_lookup(char_lookup, char_inputs))

seg_lookup = tf.get_variable(name="seg_embedding",
                             shape=[4, 3],
                             initializer=initializers.xavier_initializer())  # shape [4, 20]
embedding.append(tf.nn.embedding_lookup(seg_lookup, seg_inputs))
embed = tf.concat(embedding, axis=-1)


# Bilstm Layer  # [batch_size, num_steps, 2*lstm_dim]
used = tf.sign(tf.abs(char_inputs))
length = tf.reduce_sum(used, reduction_indices=1)
lengths = tf.cast(length, tf.int32)

dropout = tf.placeholder(dtype=tf.float32, name="Dropout")
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
output = tf.reshape(bilstm_outputs, shape=[-1, 10*2])  # 6x20, (batch_size x num_steps, lstm_units x 2)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    # embedding layer
    print(sess.run(bilstm_outputs, feed_dict={char_inputs: [[23, 24], [2, 1], [30, 41]], seg_inputs: [[0, 1], [1, 2], [1, 3]], dropout: 0.5}))
