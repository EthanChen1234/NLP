import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers


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

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print(sess.run(embedding, feed_dict={char_inputs: [[23, 24], [0, 1]], seg_inputs: [[0, 1], [1, 2]]}))
    print(tf.concat(embedding, axis=-1))