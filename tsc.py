# ### 3. TSC_Net class
# 
# The Traffic Sign Classifier (TSC) Neural Network implemention.

import tensorflow as tf
import numpy as np
import os
import sys
from tensorflow.contrib.layers import flatten

LOG_DIR = './tb_log/2xLeNet_256_8e-4_dropout'
MODEL_DIR =  './model/2xLeNet_256_8e-4_dropout'

TRAIN_DROPOUT = 0.5 
TEST_DROPOUT = 1.0


class tsc_net():
    """
    Traffic Sign Classification Network class. Derived from LeNet example.
    """
    
    def __init__(self):
        self.create_tsc_net()
        self.create_tf()
        
        self.session = tf.InteractiveSession()
        
        # Merge all summaries and write them out
        self.merged_summaries = tf.summary.merge_all()

        # Summary saving directories
        if not os.path.exists(LOG_DIR):
            os.makedirs(LOG_DIR)
        train_summary_dir = os.path.join(LOG_DIR, "train")
        test_summary_dir = os.path.join(LOG_DIR, "test")
        if not os.path.exists(train_summary_dir):
            os.makedirs(train_summary_dir)
        if not os.path.exists(test_summary_dir):
            os.makedirs(test_summary_dir)
        self.train_writer = tf.summary.FileWriter(train_summary_dir, self.session.graph)
        self.test_writer = tf.summary.FileWriter(test_summary_dir)
    
        # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
        tf.global_variables_initializer().run()

        # Add ops to save and restore all the variables.
        self.saver = tf.train.Saver()

        # Error statistics
        self.err_per_class = np.zeros([43])

    def weight_variable(self,shape,stddev=0.1):
        initial = tf.truncated_normal(shape,stddev=stddev)
        return tf.Variable(initial)

    def bias_variable(self,shape):
        initial = tf.constant(0.1, shape = shape)
        return tf.Variable(initial)

    def conv2d(self, x, W, b, strides=1):
        # Conv2D wrapper, writh bias and relu activation
        x = tf.nn.conv2d(x, W, strides=[1,strides,strides,1], padding='VALID')
        x = tf.nn.bias_add(x, b)
        #return tf.nn.relu(x)
        x = tf.nn.relu(x)
        return tf.nn.dropout(x, self.keep_prob)
    
    def create_tsc_net(self):   
        """
        Create the tensorflow based layers, cost, optimizer, etc. 
        """
        # Hyperparameters
        mu = 0
        sigma = 0.1

        # Dropout probability
        self.keep_prob = tf.placeholder(tf.float32)
        
        with tf.name_scope("input_layer"):
            # Input layer: [batch_size, 32, 32, 3] - TODO: Tensorflow doesn't support tf.float64 yet.
            self.img_in = tf.placeholder(tf.float32, [None, 32, 32, 3])
            tf.summary.histogram("input_img", self.img_in)
            
        with tf.name_scope("layer1_conv"):
            # Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
            conv1_W = self.weight_variable(shape=[5, 5, 3, 12], stddev=sigma)
            conv1_b = self.bias_variable(shape=[12])
            conv1   = self.conv2d(self.img_in, conv1_W, conv1_b)
            tf.summary.histogram("conv1_W", conv1_W)
            tf.summary.histogram("conv1_b", conv1_b)
            tf.summary.histogram("conv1", conv1)
            # Pooling. Input = 28x28x6. Output = 14x14x6.
            conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        with tf.name_scope("layer2_conv"):
            # Layer 2: Convolutional. Output = 10x10x16.
            conv2_W = self.weight_variable(shape=[5, 5, 12, 32], stddev = sigma)
            conv2_b = self.bias_variable(shape=[32])
            conv2   = self.conv2d(conv1, conv2_W, conv2_b)
            tf.summary.histogram("conv2_W", conv2_W)
            tf.summary.histogram("conv2_b", conv2_b)
            tf.summary.histogram("conv2", conv2)
            # Pooling. Input = 10x10x16. Output = 5x5x16.
            conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # Flatten. Input = 5x5x16. Output = 400.
        fc0 = flatten(conv2)

        with tf.name_scope("layer3_fc"):
            # Layer 3: Fully Connected. Input = 400. Output = 120.
            fc1_W = self.weight_variable(shape=[800, 200], stddev = sigma)
            fc1_b = self.bias_variable(shape=[200])
            fc1   = tf.matmul(fc0, fc1_W) + fc1_b
            # Activation.
            fc1    = tf.nn.relu(fc1)
            # Dropout
            fc1_dropout = tf.nn.dropout(fc1, self.keep_prob)
            tf.summary.histogram("fc1_W", fc1_W)
            tf.summary.histogram("fc1_b", fc1_b)
            tf.summary.histogram("fc1", fc1)
            tf.summary.histogram("fc1_dropout", fc1_dropout)
            
        with tf.name_scope("layer4_fc"):
            # Layer 4: Fully Connected. Input = 120. Output = 84.
            fc2_W  = self.weight_variable(shape=[200, 128], stddev = sigma)
            fc2_b  = self.bias_variable(shape=[128])
            fc2    = tf.matmul(fc1_dropout, fc2_W) + fc2_b
            # Activation.
            fc2    = tf.nn.relu(fc2)
            # Dropout
            fc2_dropout = tf.nn.dropout(fc2, self.keep_prob)
            tf.summary.histogram("fc2_W", fc2_W)
            tf.summary.histogram("fc2_b", fc2_b)
            tf.summary.histogram("fc2", fc2)
            tf.summary.histogram("fc2_dropout", fc2_dropout)
            
        with tf.name_scope("layer5_fc"):
            # Layer 5: Fully Connected. Input = 84. Output = 43.
            fc3_W  = self.weight_variable(shape=[128, 43], stddev = sigma)
            fc3_b  = self.bias_variable(shape=[43])
            self.logits = tf.matmul(fc2_dropout, fc3_W) + fc3_b
            tf.summary.histogram("fc3_W", fc3_W)
            tf.summary.histogram("fc3_b", fc3_b)
            tf.summary.histogram("logits", self.logits)
            
    def create_tf(self):
        """
        Loss/accuracy function and optimizer definition.
        """
        self.learning_rate = tf.placeholder(tf.float32)
        self.label_truth = tf.placeholder(tf.float32, [None,43])
        self.loss =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.logits, self.label_truth))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        
        #self.prediction = tf.nn.softmax(self.logits)        
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.logits, 1), 
                                    tf.argmax(self.label_truth, 1)), tf.float32))
        
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('accuracy', self.accuracy)
        
    def train(self,X,y,lr,i):
        summary, _, loss, accuracy = self.session.run(
                [self.merged_summaries, self.optimizer, self.loss, self.accuracy],
                feed_dict={
                    self.img_in: X.astype(np.float32),
                    self.label_truth: y.astype(np.float32),
                    self.keep_prob: TRAIN_DROPOUT,
                    self.learning_rate: lr
                })
        # Record summary every N batches
        if i%50 == 0:
            print('training: step {0:5d}, lr {1:8.7f}, accuracy {2:8.2f}%, loss {3:8.2f}'.format(i, lr, accuracy*100, loss))
            self.train_writer.add_summary(summary, i)

    def val(self,X,y,i,summary_on):
        summary, loss, accuracy = self.session.run(
                [self.merged_summaries, self.loss, self.accuracy],
                feed_dict={
                    self.img_in: X.astype(np.float32),
                    self.label_truth: y.astype(np.float32),
                    self.keep_prob: TEST_DROPOUT
                })
        print('validation: step {0:5d}, accuracy {1:8.2f}%, loss {2:8.2f}'.format(i, accuracy*100, loss))
        if summary_on:
            self.test_writer.add_summary(summary, i)
        
    def predict(self,X):
        logits = self.session.run(
                [self.logits],
                feed_dict={
                    self.img_in: X.astype(np.float32),
                    self.keep_prob: TEST_DROPOUT
                })
        # Predict class catogery
        for i in range(len(X)):
            print("Prediction: img {}, class {}".format(i, np.argmax(logits[0][i])))
        return logits

    def err_statistics(self,X,y):
        logits = self.session.run(
                [self.logits],
                feed_dict={
                    self.img_in: X.astype(np.float32),
                    self.label_truth: y.astype(np.float32),
                    self.keep_prob: TEST_DROPOUT
                })
        # Error counts per class/category
        for i in range(len(y)):
            if np.argmax(logits[0][i]) != np.argmax(y[i]):
                self.err_per_class[np.argmax(y[i])] += 1
        print('err statistics: err_per_class = {}'.format(self.err_per_class))
        return self.err_per_class

    def saveParam(self):
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        checkpoint_path = os.path.join(MODEL_DIR, "model.ckpt")
        filename = self.saver.save(self.session, checkpoint_path)
        print("Model saved in file: %s" % filename)

    def restoreParam(self):
        if not os.path.exists(MODEL_DIR):
            sys.exit("No such dir to restore parameters! Exiting.")
        checkpoint_path = os.path.join(MODEL_DIR, "model.ckpt")
        self.saver.restore(self.session, checkpoint_path)
        print("Model restored from file: %s" % checkpoint_path)



