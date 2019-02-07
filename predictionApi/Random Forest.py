from __future__ import print_function
import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

import tensorflow as tf
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.python.ops import resources

# Ignore all GPUs, tf random forest does not benefit from it.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

def get_historical_data():
    ''' Daily quotes from Google. Date format='yyyy-mm-dd' '''
    # import data
    df = pd.read_csv('./rsc/AAPL_data.csv')
    # shift dates forward by one day
    df.dropna()
    df['Open-Close'] = df.open - df.close
    df['High-Low'] = df.high - df.low
    return df

data = get_historical_data()

data = data.drop(['date'], 1)
data = data.dropna()

data_test = np.where(data['close'].shift(-1) > data['close'],1,-1)

data = preprocessing.scale(data)
data = pd.DataFrame(data, columns=['20_day_sma','close','ewma_12','ewma_26','high','low','obv_10','open','rsi_14','volume', 'Open-Close','High-Low'])

# Training and test data
train_start = 0
train_end = int(np.floor(0.8*len(data)))
test_start = 0
test_end = int(np.floor(0.8*len(data_test)))

# training features
data_train = np.array(data)

# Build X and y
X_train = data_train[train_start:train_end]
y_train = data_test[test_start:test_end]
X_test = data_train[train_end::]
y_test = data_test[test_end::]

num_steps = 500  # Total steps to train
batch_size = 100  # The number of samples per batch
num_classes = 2  # buy or call
num_features = 12  # number of technical indicators
num_trees = 10
max_nodes = 1000

# Input and Target data
X = tf.placeholder(tf.float32, shape=[None, num_features])
# For random forest, labels must be integers (the class id)
Y = tf.placeholder(tf.float32, shape=[None])

# Random Forest Parameters
hparams = tensor_forest.ForestHParams(num_classes=num_classes,
                                      num_features=num_features,
                                      num_trees=num_trees,
                                      max_nodes=max_nodes).fill()

# Build the Random Forest
forest_graph = tensor_forest.RandomForestGraphs(hparams)
# Get training graph and loss
train_op = forest_graph.training_graph(X, Y)
loss_op = forest_graph.training_loss(X, Y)

# Measure the accuracy
infer_op, _, _ = forest_graph.inference_graph(X)
correct_prediction = tf.equal(tf.argmax(infer_op, 1), tf.cast(Y, tf.int64))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initialize the variables (i.e. assign their default value) and forest resources
init_vars = tf.group(tf.global_variables_initializer(),
                     resources.initialize_resources(resources.shared_resources()))

# Start TensorFlow session
sess = tf.Session()

# Run the initializer
sess.run(init_vars)

# Training
for i in range(1, num_steps + 1):
    # Prepare Data

    # Shuffle training data
    shuffle_indices = np.random.permutation(np.arange(len(y_train)))
    X_train = X_train[shuffle_indices]
    y_train = y_train[shuffle_indices]

    _, l = sess.run([train_op, loss_op], feed_dict={X: X_train, Y: y_train})
    if i % 50 == 0 or i == 1:
        acc = sess.run(accuracy_op, feed_dict={X: X_train, Y: y_train})
        print('Step %i, Loss: %f, Acc: %f' % (i, l, acc))

# Test Model
test_x, test_y = X_test, y_test
print("Test Accuracy:", sess.run(accuracy_op, feed_dict={X: test_x, Y: test_y}))