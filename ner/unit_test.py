import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers


# embedding layer  # [1, num_steps, embedding size], embedding size = 100+20, num_steps =
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


# # Bilstm Layer
used = tf.sign(tf.abs(char_inputs))
length = tf.reduce_sum(used, reduction_indices=1)
lengths = tf.cast(length, tf.int32)

dropout = tf.placeholder(dtype=tf.float32, name="Dropout")
model_inputs = tf.nn.dropout(embed, rate=1-dropout)
model_outputs = biLSTM_layer(model_inputs, 100, lengths)


with tf.Session() as sess:
    tf.global_variables_initializer().run()
    # embedding layer
    # sess.run(embed, feed_dict={char_inputs: [[23, 24], [0, 1]], seg_inputs: [[0, 1], [1, 2]]})

    print(sess.run(lengths, feed_dict={char_inputs: [[23, 24], [0, 1]], seg_inputs: [[0, 1], [1, 2]], dropout: 0.5}))
