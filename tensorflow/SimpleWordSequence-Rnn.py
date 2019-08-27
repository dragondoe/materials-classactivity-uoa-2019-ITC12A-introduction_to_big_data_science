# Lab 12 RNN
import tensorflow as tf
import numpy as np
tf.set_random_seed(777)  # reproducibility

strings = ["Big ", "Data ", "Deep ", "Learning ", "Course "]
# Teach words: Big Data Big Deep Learning -> Data Big Deep Learning Course
x_data = [[0, 1, 0, 2, 3]]   # Big Data Big Deep Learning
x_one_hot = [[[1, 0, 0, 0, 0],   # Big 0
              [0, 1, 0, 0, 0],   # Data 1
              [1, 0, 0, 0, 0],   # Big 0
              [0, 0, 1, 0, 0],   # Deep 2
              [0, 0, 0, 1, 0]]]  # Learning 3              
              
x_t_data = [[0, 1, 0, 2, 3]]   # Big Data Big Deep Learning
x_t_one_hot = [[[1, 0, 0, 0, 0],   # Big 0
              [0, 1, 0, 0, 0],   # Data 1
              [1, 0, 0, 0, 0],   # Big 0
              [0, 0, 1, 0, 0],   # Deep 2
              [0, 0, 0, 1, 0]]]  # Learning 3
y_data = [[1, 0, 2, 3, 4]]    # Data Big Deep Learning Course

num_classes = 5
input_dim = 5  # one-hot size
hidden_size = 5  # output from the LSTM. 5 to directly predict one-hot
batch_size = 1   # one sentence
sequence_length = 5  # |Data Big Deep Learning Course| == 5
learning_rate = 0.1

X = tf.placeholder(
    tf.float32, [None, sequence_length, input_dim])  # X one-hot
Y = tf.placeholder(tf.int32, [None, sequence_length])  # Y label

cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
#cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size)
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(
    cell, X, initial_state=initial_state, dtype=tf.float32)

# FC layer
X_for_fc = tf.reshape(outputs, [-1, hidden_size])
# fc_w = tf.get_variable("fc_w", [hidden_size, num_classes])
# fc_b = tf.get_variable("fc_b", [num_classes])
# outputs = tf.matmul(X_for_fc, fc_w) + fc_b
outputs = tf.contrib.layers.fully_connected(
    inputs=X_for_fc, num_outputs=num_classes, activation_fn=None)

# reshape out for sequence_loss
outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])

weights = tf.ones([batch_size, sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(
    logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

prediction = tf.argmax(outputs, axis=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(50):
        l, _ = sess.run([loss, train], feed_dict={X: x_one_hot, Y: y_data})
        result = sess.run(prediction, feed_dict={X: x_one_hot})
        print(i, "loss:", l, "prediction: ", result, "true Y: ", y_data)

        # print char using dic
        # result_str = [idx2char[c] for c in np.squeeze(result)]
        result_str = [strings[c] for c in np.squeeze(result)]
        print("\tPrediction str: ", ''.join(result_str))
    print("*************** After Training, Test...!! *****")
    result2 = sess.run(prediction, feed_dict={X: x_t_one_hot})
    print(i, "loss:", l, "prediction: ", result, "true Y: ", y_data)
    # print char using dic
    # result2_str = [idx2char[c] for c in np.squeeze(result2)]
    result2_str = [strings[c] for c in np.squeeze(result2)]
    print("\tPrediction str for test: ", ''.join(result2_str))

