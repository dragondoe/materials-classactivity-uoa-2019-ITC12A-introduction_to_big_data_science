'''
A Multilayer Perceptron implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

import tensorflow as tf
import numpy
# import input_data

batch_size = 50
filename_queue_train = tf.train.string_input_producer(["drama-economy-train-width-1000.csv"])
filename_queue_test = tf.train.string_input_producer(["drama-economy-test-width-1000.csv"])

reader1 = tf.TextLineReader()
key1, trainValue = reader1.read(filename_queue_train)

reader2 = tf.TextLineReader()
key2, testValue = reader2.read(filename_queue_test)

# Default values, in case of empty columns. Also specifies the type of the
# decoded result.

#default = [[0.0] for x in range(786)]
default = [[0.0] for x in range(1002)]

##################  For Training Data  ############################
line_train = tf.decode_csv(trainValue, record_defaults=default, field_delim=",", name=None)
train_label_pack = tf.stack([line_train[0], line_train[1]])
train_feature_pack = tf.stack(list(line_train[2:]))
train_label_batch, train_feature_batch = tf.train.batch([train_label_pack, train_feature_pack], batch_size = batch_size, num_threads=1)

##################  For Test Data  ############################
line_test = tf.decode_csv(testValue, record_defaults=default)
test_label_pack = tf.stack([line_test[0], line_test[1]])
test_feature_pack = tf.stack(list(line_test[2:]))
test_label_batch, test_feature_batch = tf.train.batch([test_label_pack, test_feature_pack], batch_size = 400, num_threads=1)

####################################################################

# Parameters
#learning_rate = 0.001
learning_rate = 0.03
training_epochs = 60
#batch_size = 20
display_step = 1
num_examples = 400

# Network Parameters
n_hidden_1 = 500 # 1st layer num features
n_hidden_2 = 500 # 2nd layer num features
n_input = 1000  # Size of Feature Vector
n_classes = 2 # +1 / 0

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])
# x = tf.placeholder(tf.float32, [None, n_input])
# y = tf.placeholder(tf.float32, [None, n_classes])

# Create model
def multilayer_perceptron(_X, _weights, _biases):
    layer_1 = tf.sigmoid(tf.add(tf.matmul(_X, _weights['h1']), _biases['b1'])) #Hidden layer with Sigmoid activation
    layer_2 = tf.sigmoid(tf.add(tf.matmul(layer_1, _weights['h2']), _biases['b2'])) #Hidden layer with Sigmoid activation
    return tf.matmul(layer_2, _weights['out']) + _biases['out']

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y)) # Softmax loss -- Old version
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)) # Softmax loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer

# Initializing the variables
#init = tf.initialize_all_variables()
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
  sess.run(init)
  coord = tf.train.Coordinator() 
  threads = tf.train.start_queue_runners(sess, coord=coord)
  #tf.train.start_queue_runners(sess)

  # Training Cycle
  try:
   #while not coord.should_stop():
      for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(num_examples/batch_size)
        # Loop over all batches
        for i in range (total_batch): 
          # print ("Optimizing one batch: ", i)
          ### Read A Batch
          train_label, train_feature = sess.run([train_label_batch, train_feature_batch])
         #test_label, test_feature = sess.run([test_label_batch, test_feature_batch])
          ### End of Read A Batch
          sess.run(optimizer, feed_dict={x: train_feature, y: train_label})
          # Compute average loss
          avg_cost += sess.run(cost, feed_dict={x: train_feature, y: train_label}) / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
          print( "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

      print ("Optimization Finished!")

      # Test model
      correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
      # Calculate accuracy
      test_label, test_feature = sess.run([test_label_batch, test_feature_batch])
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
      print( "Accuracy:", accuracy.eval({x: test_feature, y: test_label}))

  except tf.erroes.OutOfRangeError:
    #logging.warning('error occured: {}'.format(e))
    print('Done training -- epoch limit reached.. ')

  except tf.erroes.CancelledError:
    print('Cancelled Error.. ')

  finally:
    coord.request_stop()
  coord.join(threads)
    #sess.colse()
