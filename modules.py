from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

n_input = 28 #MNIST data input with image shape 28*28
n_steps = 28
n_hidden = 128
n_classes = 10

#tf Graph input
x = tf.placeholder("float",[None,n_steps,n_input])
y = tf.placeholder("float",[None,n_classes])
weights = {
          'out':tf.Variable(tf.random_normal([n_hidden, n_classes]))
          }
biases = {
          'out':tf.Variable(tf.random_normal([n_classes]))
          }
