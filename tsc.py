
# ### 2. Dataset class
# - Retrieving pre-processed pickle file. 
# - Function to load train/val/test batch.

# In[6]:

import tensorflow as tf
import numpy as np
import pickle
import os

class dataset():
    """
    Dataset class to retrieve pickle file, and
    provide function to load train/val batch, test data.
    """
    def __init__(self):
        self.train_pointer = 0

    def reset_ptr(self):
        """
        Reset pointers for new epoch.
        """
        self.train_pointer = 0
        
    def open_dataset(self, file):
        self.f = open(file, 'rb')
        data = pickle.load(self.f)
        self.train_dataset = data['train_dataset']
        self.train_labels = data['train_labels']
        self.valid_dataset = data['valid_dataset']
        self.valid_labels = data['valid_labels']
        self.test_dataset = data['test_dataset']
        self.test_labels = data['test_labels']
        
        self.train_dataset_size = self.train_dataset.shape[0]
        #print("self.train_dataset_size =", self.train_dataset_size)
        
    def close_dataset(self):
        self.f.close()
        
    def load_train_batch(self, batch_size):
        if self.train_pointer + batch_size >= self.train_dataset_size:
            # At the end of training dataset items, less than batch_size requested
            batch = self.train_dataset_size
        else:
            # Get next batch_size out
            batch = self.train_pointer + batch_size
            
        X_out = self.train_dataset[self.train_pointer:batch,:,:,:]
        y_out = self.train_labels[self.train_pointer:batch]
        self.train_pointer = batch
        return X_out, y_out, self.train_pointer>=self.train_dataset_size
    
    def load_valid_data(self):
        # Test data doesn't need to be loaded in batches.
        return self.valid_dataset, self.valid_labels
    
    def load_test_data(self):
        # Test data doesn't need to be loaded in batches.
        return self.test_dataset, self.test_labels

    
# Load preprocessed data
file = 'preprocessed.pickle'
d = dataset()
d.open_dataset(file)
X_train_batch_1, y_train_batch_1, _ = d.load_train_batch(128)
for i in range(306): # Should report 40 training data, the last batch
    X_train_batch, y_train_batch, runout = d.load_train_batch(128)
    X_valid_batch, y_valid_batch = d.load_valid_data()
    X_test_dataset,  y_test_dataset  = d.load_test_data()

# Debug prints
#print("train_batch[2]=", X_train_batch[2])
#print("train_label[2]=", y_train_batch[2])
print("train_batch.shape = ", X_train_batch.shape)
print("max value in train_batch:", np.max(X_train_batch))
print("min value in train_batch:", np.min(X_train_batch))
print("mean value in train_reset:", np.mean(X_train_batch))
print("train_batch pointer = ", d.train_pointer)
assert runout==True, "ASSERT: Cannot read out all data in training set!!!"

print("validation_batch.shape = ", X_valid_batch.shape)
#print("valid_batch[2]=", X_valid_batch[2])
#print("valid_label[2]=", y_valid_batch[2])
print("max value in validation_batch:", np.max(X_valid_batch))
print("min value in validation_batch:", np.min(X_valid_batch))
print("mean value in validation_batch:", np.mean(X_valid_batch))

print("test_dataset.shape = ", X_test_dataset.shape)
#print("test_dataset[2]=", X_test_dataset[2])
#print("test_label[2]=", y_test_dataset[2])
print("max value in test_dataset:", np.max(X_test_dataset))
print("min value in test_dataset:", np.min(X_test_dataset))
print("mean value in test_dataset:", np.mean(X_test_dataset))

# Reset pointers test
print("Testing dataset.reset_ptr()...")
d.reset_ptr()
X_train_batch_reset, y_train_batch_reset, _ = d.load_train_batch(128)

print("train_batch_reset.shape = ", X_train_batch_reset.shape)
print("max value in train_batch_reset:", np.max(X_train_batch_reset))
print("min value in train_batch_reset:", np.min(X_train_batch_reset))
print("mean value in train_batch_reset:", np.mean(X_train_batch_reset))
print("train_batch pointer = ", d.train_pointer)

assert np.array_equal(X_train_batch_1, X_train_batch_reset), "dataset class reset_ptr() doesn't work!"
assert np.array_equal(y_train_batch_1, y_train_batch_reset), "dataset class reset_ptr() doesn't work!"

d.close_dataset()



# ### 3. TSC_Net class
# 
# The Traffic Sign Classifier (TSC) Neural Network implemention.

# In[7]:

import os
import sys
from tensorflow.contrib.layers import flatten

LOG_DIR = './tensorboard_log/LeNet'
MODEL_DIR =  './model'

TRAIN_DROPOUT = 0.5 
TEST_DROPOUT = 1.0
LEARNING_RATE = 1e-6


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
        self.train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, "train"), self.session.graph)
        self.test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, "test"))
    
        # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
        tf.global_variables_initializer().run()

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
        return tf.nn.relu(x)
    
    def create_tsc_net(self):   
        """
        Create the tensorflow based layers, cost, optimizer, etc. 
        """
        # Hyperparameters
        mu = 0
        sigma = 0.1
        
        with tf.name_scope("input_layer"):
            # Input layer: [batch_size, 32, 32, 3] - TODO: Tensorflow doesn't support tf.float64 yet.
            self.img_in = tf.placeholder(tf.float32, [None, 32, 32, 3])
            tf.summary.histogram("input_img", self.img_in)
            
        with tf.name_scope("layer1_conv"):
            # Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
            conv1_W = self.weight_variable(shape=[5, 5, 3, 6], stddev=sigma)
            conv1_b = self.bias_variable(shape=[6])
            conv1   = self.conv2d(self.img_in, conv1_W, conv1_b)
            tf.summary.histogram("conv1_W", conv1_W)
            tf.summary.histogram("conv1_b", conv1_b)
            tf.summary.histogram("conv1", conv1)
            # Pooling. Input = 28x28x6. Output = 14x14x6.
            conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        with tf.name_scope("layer2_conv"):
            # Layer 2: Convolutional. Output = 10x10x16.
            conv2_W = self.weight_variable(shape=[5, 5, 6, 16], stddev = sigma)
            conv2_b = self.bias_variable(shape=[16])
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
            fc1_W = self.weight_variable(shape=[400, 120], stddev = sigma)
            fc1_b = self.bias_variable(shape=[120])
            fc1   = tf.matmul(fc0, fc1_W) + fc1_b
            # Activation.
            fc1    = tf.nn.relu(fc1)
            tf.summary.histogram("fc1_W", fc1_W)
            tf.summary.histogram("fc1_b", fc1_b)
            tf.summary.histogram("fc1", fc1)
            
        with tf.name_scope("layer4_fc"):
            # Layer 4: Fully Connected. Input = 120. Output = 84.
            fc2_W  = self.weight_variable(shape=[120, 84], stddev = sigma)
            fc2_b  = self.bias_variable(shape=[84])
            fc2    = tf.matmul(fc1, fc2_W) + fc2_b
            # Activation.
            fc2    = tf.nn.relu(fc2)
            tf.summary.histogram("fc2_W", fc2_W)
            tf.summary.histogram("fc2_b", fc2_b)
            tf.summary.histogram("fc2", fc2)
            
        with tf.name_scope("layer5_fc"):
            # Layer 5: Fully Connected. Input = 84. Output = 43.
            fc3_W  = self.weight_variable(shape=[84, 43], stddev = sigma)
            fc3_b  = self.bias_variable(shape=[43])
            self.logits = tf.matmul(fc2, fc3_W) + fc3_b
            tf.summary.histogram("fc3_W", fc3_W)
            tf.summary.histogram("fc3_b", fc3_b)
            tf.summary.histogram("logits", self.logits)
            
    def create_tf(self):
        """
        Loss/accuracy function and optimizer definition.
        """
        self.label_truth = tf.placeholder(tf.float32, [None,43])
        self.loss =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.logits, self.label_truth))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(self.loss)
        
        #self.prediction = tf.nn.softmax(self.logits)        
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.logits, 1), 
                                    tf.argmax(self.label_truth, 1)), tf.float32))
        
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('accuracy', self.accuracy)
        
    def train(self,X,y,i):
        summary, _, loss = self.session.run(
                [self.merged_summaries, self.optimizer, self.loss],
                feed_dict={
                    self.img_in: X.astype(np.float32),
                    self.label_truth: y.astype(np.float32)
                })
        # Record summary every N batches
        if i%10 == 0:
            self.train_writer.add_summary(summary, i)

    def val(self,X,y,i):
        summary, loss, accuracy = self.session.run(
                [self.merged_summaries, self.loss, self.accuracy],
                feed_dict={
                    self.img_in: X.astype(np.float32),
                    self.label_truth: y.astype(np.float32)
                })
        print('val: step {0:5d}, accuracy {1:8.2f}%, loss {2:8.2f}'.format(i, accuracy*100, loss))
        self.test_writer.add_summary(summary, i)
        
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


# ### 4. Stitching everthing together

# In[8]:

def main():
    mynet = tsc_net()
    data = dataset()
    
    file = 'preprocessed.pickle'

    data.open_dataset(file)

    for i in range(100):
        X_train_batch, y_train_batch, end_of_train_dataset = data.load_train_batch(128)
        if i==0 :
            print("X_train_batch.shape = ", np.shape(X_train_batch))
            print("X_train_batch.dtype = ", X_train_batch.dtype)
            print("y_train_batch.shape = ", np.shape(y_train_batch))
            print("y_train_batch.dtype = ", y_train_batch.dtype)
        mynet.train(X_train_batch, y_train_batch, i)
        mynet.val(X_train_batch, y_train_batch, i)

    X_valid_dataset, y_valid_dataset = data.load_valid_data()
    mynet.val(X_valid_dataset, y_valid_dataset, 1)
    X_test_dataset,  y_test_dataset  = data.load_test_data()
    mynet.val(X_test_dataset, y_test_dataset, 1)

    # Reset pointers test
    print("Testing dataset.reset_ptr()...")
    data.reset_ptr()
    X_train_batch, y_train_batch, end_of_train_dataset = data.load_train_batch(128)
    mynet.train(X_train_batch, y_train_batch, 1)
    data.close_dataset()

if __name__ == '__main__':
    main()


