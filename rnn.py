def RNN(x, weights, biases):
    x = tf.unstack(x, n_steps, 1)
    
    #Define a LSTM cell with Tensorflow
    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    
    #Get lstm cell output
    output, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    
    #Linear activation using rnn inner loop last output
    return tf.matmul(output[-1], weights['out']) + biases['out']
    
 pred = RNN(x, weights, biases)
 
 #Define loss and optimizer
 cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
 optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
 
 #Evaluate model
 correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
 accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
 
 #Initializing the variables
 init = tf.global_variables_initializer()
