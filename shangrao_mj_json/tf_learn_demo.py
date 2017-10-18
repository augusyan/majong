# -*- coding:utf-8 -*-
"""
@author : Augus Yan
@file : tf_learn_demo.py
@time : 2017/10/11 9:48
@function : 
"""
from __future__ import absolute_import, division, print_function
import tflearn
import tflearn.datasets.mnist as mnist
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import tflearn as tfl
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
import csv
import os
# 封装一些相关设置


class Model:

    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            # input place holders
            self.X = tf.placeholder(tf.float32, [None, 784])
            # img 28x28x1 (black/white)
            X_img = tf.reshape(self.X, [-1, 28, 28, 1])
            self.Y = tf.placeholder(tf.float32, [None, 10])
            # L1 ImgIn shape=(?, 28, 28, 1)
            # W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))


def predict(self, x_test, keep_prop=1.0):
    return self.sess.run(self.logits, feed_dict={self.X: x_test, self.keep_prob: keep_prop})


def get_accuracy(self, x_test, y_test, keep_prop=1.0):
    return self.sess.run(self.accuracy,  feed_dict={self.X: x_test, self.Y: y_test, self.keep_prob: keep_prop})


def train(self, x_data, y_data, keep_prop=0.7):
    return self.sess.run([self.cost, self.optimizer], feed_dict={
       self.X: x_data, self.Y: y_data, self.keep_prob: keep_prop})


"""class Model(object):
    def __init__(self, is_training, config, input_, data_type):
        self._input = input_
        batch_size = input_.batch_size
        num_steps = input_.num_steps
        size = config.hidden_size
        vocab_size = config.vocab_size
        # 一些相关设置
        output = tf.reshape(tf.concat(1, outputs), [-1, size])
        softmax_w = tf.get_variable("softmax_w", [size, vocab_size], dtype=data_type())
        softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
        logits = tf.matmul(output, softmax_w) + softmax_b
        loss = tf.nn.seq2seq.sequence_loss_by_example([logits], [tf.reshape(input_.targets, [-1])],
                                                      [tf.ones([batch_size * num_steps], dtype=data_type())])
        self._cost = cost = tf.reduce_sum(loss) / batch_size
"""
# 线性回归实现
"""

a = tf.zeros(shape=[1, 2])
x_train = [1, 2, 3]
y_train = [1, 2, 3]

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
_hypothesis = x_train * W + b
_op = tf.multiply(x, y)

cost = tf.reduce_mean(tf.square(_hypothesis - y_train))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

# Fit the line for step in range(2001):
cost_val, W_val, b_val, _ = sess.run([cost, W, b, train],
feed_dict={X: [1, 2, 3], Y: [1, 2, 3]})
if step % 20 == 0:
print(step, cost_val, W_val, b_val)
"""
# 实现LR简单版
"""
x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]
# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([2, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
# Hypothesis using sigmoid: tf.div(1., 1. + tf.exp(tf.matmul(X, W)))
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)
# cost/loss function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# Accuracy computation
# True if hypothesis>0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# Launch graph
with tf.Session() as sess:
   # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())
    for step in range(10001):
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print(step, cost_val)

    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data})
    print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)
"""
# 实现LR读取文件版
"""
xy = np.loadtxt('D:\\Planet Saving Plan\\Tensorflow\\DeepLearningZeroToAll-master\\data-03-diabetes.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 8])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([8, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis using sigmoid: tf.div(1., 1. + tf.exp(tf.matmul(X, W)))
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)
# cost/loss function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# Accuracy computation True if hypothesis>0.5 else False

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# Launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    feed = {X: x_data, Y: y_data}
    for step in range(10001):
        sess.run(train, feed_dict=feed)
        if step % 200 == 0:
            print(step, sess.run(cost, feed_dict=feed))
    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict=feed)
    print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)
"""
# softmax简单版
"""x_data = [[1, 2, 1, 1], [2, 1, 3, 2], [3, 1, 3, 4], [4, 1, 5, 5], [1, 7, 5, 5], [1, 2, 5, 6], [1, 6, 6, 6], [1, 7, 7, 7]]
y_data = [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]]

X = tf.placeholder("float", [None, 4])
Y = tf.placeholder("float", [None, 3])
nb_classes = 3

W = tf.Variable(tf.random_normal([4, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

# tf.nn.softmax computes softmax activations
# softmax = exp(logits) / reduce_sum(exp(logits), dim)
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

# Cross entropy cost/loss
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}))
"""

# mnist batch 操作
"""
# Check out https://www.tensorflow.org/get_started/mnist/beginners for
# more information about the mnist dataset
mnist = input_data.read_data_sets("D://datasets//MNIST", one_hot=True)

feature_dim = 784
nb_classes = 10
training_epochs = 15
batch_size = 100
learning_rate = 0.1

# MNIST data image of shape 28 * 28 = 784
X = tf.placeholder(tf.float32, [None, feature_dim])
# 0 - 9 digits recognition = 10 classes
Y = tf.placeholder(tf.float32, [None, nb_classes])

# weights & bias for nn layers
W = tf.Variable(tf.random_normal([784, 10]))
b = tf.Variable(tf.random_normal([10]))

hypothesis = tf.matmul(X, W) + b
# define cost/loss & optimizer
# cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer =tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# initialize
with tf.Session() as sess:
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # train my model
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            feed_dict = {X: batch_xs, Y: batch_ys}
            c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
            avg_cost += c / total_batch
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
print('Learning Finished!')
# Test model and check accuracy
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))
"""
# Get one and predict
"""
r = random.randint(0, mnist.test.num_examples - 1)
print("Label:", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
print("Prediction:", sess.run(tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r + 1]}))
plt.imshow(mnist.test.images[r:r + 1].reshape(28, 28), cmap='Greys', interpolation='nearest')
plt.show()
"""

# XOR test error version
"""
x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
W = tf.Variable(tf.random_normal([2, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis using sigmoid: tf.div(1., 1. + tf.exp(tf.matmul(X, W)))
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

# cost/loss function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Accuracy computation
# True if hypothesis>0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# Launch graph
with tf.Session() as sess:
   # Initialize TensorFlow variables
   sess.run(tf.global_variables_initializer())

   for step in range(10001):
       sess.run(train, feed_dict={X: x_data, Y: y_data})
       if step % 100 == 0:
           print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))

   # Accuracy report
   h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data})
   print("\nHypothesis: ", h, "\nCorrect: ", c, "\nAccuracy: ", a)
"""
"""# mnist cnn 版，带模型保存
mnist = input_data.read_data_sets("D:\\datasets\\MNIST", one_hot=True)
# 下载并加载mnist数据
x = tf.placeholder(tf.float32, [None, 784], name='x')  # 输入的数据占位符
y_actual = tf.placeholder(tf.float32, shape=[None, 10], name='y_actual')


# 定义一个函数，用于初始化所有的权值 W
def weight_variable(shape, name_input):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name_input)


# 定义一个函数，用于初始化所有的偏置项 b
def bias_variable(shape, name_input):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name_input)


# 定义一个函数，用于构建卷积层
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# 定义一个函数，用于构建池化层
def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 构建网络
x_image = tf.reshape(x, [-1, 28, 28, 1])  # 转换输入数据shape,以便于用于网络中
W_conv1 = weight_variable([5, 5, 1, 32], name_input='W_conv1')
b_conv1 = bias_variable([32], name_input='b_conv1b_conv1')
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # 第一个卷积层
h_pool1 = max_pool(h_conv1)  # 第一个池化层

W_conv2 = weight_variable([5, 5, 32, 64], name_input='W_conv2')
b_conv2 = bias_variable([64], name_input='b_conv2')
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # 第二个卷积层
h_pool2 = max_pool(h_conv2)  # 第二个池化层

W_fc1 = weight_variable([7 * 7 * 64, 1024], name_input='W_fc1')
b_fc1 = bias_variable([1024], name_input='b_fc1')
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])  # reshape成向量
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  # 第一个全连接层

keep_prob = tf.placeholder("float", name='keep_prob')
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)  # dropout层

W_fc2 = weight_variable([1024, 10], name_input='W_fc2')
b_fc2 = bias_variable([10], name_input='b_fc2')

y_predict = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)  # softmax层

cross_entropy = -tf.reduce_sum(y_actual * tf.log(y_predict))  # 交叉熵
train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(cross_entropy)  # 梯度下降法
correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_actual, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))  # 精确度计算
saver = tf.train.Saver()  # 保存模型
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
for i in range(20001):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:  # 训练100次，验证一次
        train_acc = accuracy.eval(feed_dict={x: batch[0], y_actual: batch[1], keep_prob: 1.0})
        print('step', i, 'training accuracy', train_acc)
        train_step.run(feed_dict={x: batch[0], y_actual: batch[1], keep_prob: 0.5})
test_acc = accuracy.eval(feed_dict={x: mnist.test.images, y_actual: mnist.test.labels, keep_prob: 1.0})
print("test accuracy", test_acc)
saver_path = saver.save(sess, "D:\\datasets\\MNIST\\model.ckpt")  # 将模型保存到save/model.ckpt文件
print("Model saved in file:", saver_path)
sess.close()"""

# tflearn版的模型读取存储
"""
# An example showing how to save/restore models and retrieve weights.
# MNIST Data
X, Y, testX, testY = mnist.load_data(one_hot=True)

# Model
input_layer = tflearn.input_data(shape=[None, 784], name='input')
dense1 = tflearn.fully_connected(input_layer, 128, name='dense1')
dense2 = tflearn.fully_connected(dense1, 256, name='dense2')
softmax = tflearn.fully_connected(dense2, 10, activation='softmax')
regression = tflearn.regression(softmax, optimizer='adam',
                                learning_rate=0.001,
                                loss='categorical_crossentropy')

# Define classifier, with model checkpoint (autosave)
model = tflearn.DNN(regression, checkpoint_path='model.tfl.ckpt')

# Train model, with model checkpoint every epoch and every 200 training steps.
model.fit(X, Y, n_epoch=1,
          validation_set=(testX, testY),
          show_metric=True,
          snapshot_epoch=True, # Snapshot (save & evaluate) model every epoch.
          snapshot_step=500, # Snapshot (save & evalaute) model every 500 steps.
          run_id='model_and_weights')


# ---------------------
# Save and load a model
# ---------------------

# Manually save model
model.save("model.tfl")

# Load a model
model.load("model.tfl")

# Or Load a model from auto-generated checkpoint
# >> model.load("model.tfl.ckpt-500")

# Resume training
model.fit(X, Y, n_epoch=1,
          validation_set=(testX, testY),
          show_metric=True,
          snapshot_epoch=True,
          run_id='model_and_weights')

# ------------------
# Retrieving weights
# ------------------
# Retrieve a layer weights, by layer name:
dense1_vars = tflearn.variables.get_layer_variables_by_name('dense1')
# Get a variable's value, using model `get_weights` method:
print("Dense1 layer weights:")
print(model.get_weights(dense1_vars[0]))
# Or using generic tflearn function:
print("Dense1 layer biases:")
with model.session.as_default():
    print(tflearn.variables.get_value(dense1_vars[1]))

# It is also possible to retrieve a layer weights through its attributes `W`
# and `b` (if available).
# Get variable's value, using model `get_weights` method:
print("Dense2 layer weights:")
print(model.get_weights(dense2.W))
# Or using generic tflearn function:
print("Dense2 layer biases:")
with model.session.as_default():
    print(tflearn.variables.get_value(dense2.b))

# dropout (keep_prob) rate  0.7 on training, but should be 1 for testing
"""
"""keep_prob = tf.placeholder(tf.float32)

W1 = tf.get_variable("W1", shape=[784, 512])
b1 = tf.Variable(tf.random_normal([512]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

W2 = tf.get_variable("W2", shape=[512, 512])
b2 = tf.Variable(tf.random_normal([512]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

# train my model
for epoch in range(training_epochs):
    ...
for i in range(total_batch):
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    feed_dict = {X: batch_xs, Y: batch_ys, keep_prob: 0.7}

# Test model and check accuracy
[...]
print('Accuracy:', sess.run(accuracy, feed_dict={
    X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1}))
"""

# 从csv中读取数据并处理
"""os.chdir("/Users/apple/Desktop/")
print(os.getcwd())


def read_data(file_queue):
    reader = tf.TextLineReader(skip_header_lines=0)
    key, value = reader.read(queue=file_queue)
    defaults = list([0.] for i in range(17))
    input_Idq_s_1, input_Idq_s_2, input_Edq_s_1, input_Edq_s_2, input_MagV_s, input_Idq_m_1, input_Idq_m_2, \
    input_Edq_m_1, input_Edq_m_2, input_MagV_m, input_Vdc, input_Pt, output_Vdq_s_1, output_Vdq_s_2, \
    output_Vdq_m_1, output_Vdq_m_2, rand = tf.decode_csv(records=value, record_defaults=defaults)
    input_list = list([input_Idq_s_1, input_Idq_s_2, input_Edq_s_1, input_Edq_s_2, input_MagV_s, input_Idq_m_1, \
                       input_Idq_m_2, input_Edq_m_1, input_Edq_m_2, input_MagV_m, input_Vdc, input_Pt])
    output_list = list([output_Vdq_s_1, output_Vdq_s_2, output_Vdq_m_1, output_Vdq_m_2])
    input_minimum = list([26.9223227068843, -98.4780345017635, 94.2182270883746, -2547.04028098514, 188.434752223462,
                          -3074.24987628734, -80.0663030792083, 94.3688439437233, -4294.32895768398, 188.737903943159,
                          11.3363750371760, \
                          14.5698874167718])
    input_maximum = list([338.471317085834, 132.425043875557, 8178.66679759072, 3962.04092754847, 14196.4064943722, \
                          1551.68716264095, 75.9558804677249, 8170.27398930453, 4158.28883844735, 14515.0865307378,
                          20098.8974807069, \
                          12123066.0740678])
    output_minimum = list([-105.675264765919, -3839.50483890428, -9675.45018951087, -3704.86493155417])
    output_maximum = list([10981.3824330706, 5832.43660916112, 105.536592510222, 8641.08573453797])
    input_normalization = list(map(lambda x, min_x, max_x: (x - min_x) / (max_x - min_x), input_list, \
                                   input_minimum, input_maximum))
    output_normalization = list(map(lambda x, min_x, max_x: (x - min_x) / (max_x - min_x), output_list, \
                                    output_minimum, output_maximum))
    return input_normalization, output_normalization[3]


def create_pipeline(filename, batch_size, num_epochs=None):
    file_queue = tf.train.string_input_producer(string_tensor=[filename],
                                                num_epochs=num_epochs)
    example, label = read_data(file_queue=file_queue)
    min_after_dequeue = 1000
    capacity = min_after_dequeue + batch_size
    example_batch, label_batch = tf.train.shuffle_batch(
        tensors=[example, label],
        batch_size=batch_size,
        capacity=capacity,
        min_after_dequeue=min_after_dequeue
    )
    return example_batch, label_batch


def relu(x, alpha=0., max_value=None):
    negative_part = tf.nn.relu(-x)
    x = tf.nn.relu(x)
    if max_value is not None:
        x = tf.clip_by_value(x, tf.cast(0., dtype=tf.float32),
                             tf.cast(max_value, dtype=tf.float32))
    x -= tf.constant(alpha, dtype=tf.float32) * negative_part
    return x


def add_layer(inputs, in_size, out_size, activation_function=None, layer_name='Layer'):
    with tf.name_scope(layer_name):
        with tf.name_scope('Weights'):
            Weights = tf.Variable(initial_value=tf.random_normal([in_size, out_size]))
        with tf.name_scope('Biases'):
            biases = tf.Variable(initial_value=tf.zeros(shape=[1, out_size]) + 0.1)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if activation_function is None:
            outputs = Wx_plus_b
        elif activation_function == tf.nn.relu:
            outputs = relu(Wx_plus_b, alpha=0.05)
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs, Weights, biases


x_train_batch, y_train_batch = create_pipeline(filename='m3c_data_train.csv', batch_size=50, num_epochs=1000)
x_test, y_test = create_pipeline(filename='m3c_data_test.csv', batch_size=60)
global_step = tf.Variable(initial_value=0, trainable=False)
# learning_rate = tf.constant(1e-3, dtype=tf.float32)
learning_rate = tf.train.exponential_decay(1e-4, global_step, 1e4, 1e-5)

with tf.name_scope('InputLayer'):
    x = tf.placeholder(dtype=tf.float32, shape=[None, 12])
    y = tf.placeholder(dtype=tf.float32, shape=[None])

hidden_0, Weights_hidden_0, biases_hidden_0 = add_layer(inputs=x, in_size=12, out_size=5,
                                                        activation_function=tf.nn.relu, layer_name='HiddenLayer0')
hidden_1, Weights_hidden_1, biases_hidden_1 = add_layer(inputs=hidden_0, in_size=5, out_size=5,
                                                        activation_function=tf.nn.relu, layer_name='HiddenLayer1')
prediction, Weights_prediction, biases_prediction = add_layer(inputs=hidden_1, in_size=5, out_size=1,
                                                              activation_function=None, layer_name='OutputLayer')

with tf.name_scope('Loss'):
    loss = tf.reduce_mean(tf.reduce_sum(input_tensor=tf.square(y - prediction), reduction_indices=[1]))
tf.summary.scalar(name='Loss', tensor=loss)

with tf.name_scope('Train'):
    train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss=loss, \
                                                                                         global_step=global_step)

correct_prediction = tf.equal(tf.argmax(input=prediction, axis=1), tf.cast(y, tf.int64))
accuracy = tf.reduce_mean(input_tensor=tf.cast(correct_prediction, tf.float32))
tf.summary.scalar(name='Accuracy', tensor=accuracy)

init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
merged_summary = tf.summary.merge_all()

sess = tf.Session()
train_writer = tf.summary.FileWriter(logdir='logs/train', graph=sess.graph)
test_writer = tf.summary.FileWriter(logdir='logs/test', graph=sess.graph)
sess.run(init)

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

try:
    print("Training: ")
    count = 0
    curr_x_test_batch, curr_y_test_batch = sess.run([x_test, y_test])

    while not coord.should_stop():
        count += 1
        curr_x_train_batch, curr_y_train_batch = sess.run([x_train_batch, y_train_batch])
        sess.run(train_step, feed_dict={
            x: curr_x_train_batch, y: curr_y_train_batch
        })
        lr = sess.run(learning_rate)
        loss_, summary = sess.run([loss, merged_summary], feed_dict={
            x: curr_x_train_batch, y: curr_y_train_batch
        })
        train_writer.add_summary(summary, count)
        loss_, test_acc, test_summary = sess.run([loss, accuracy, merged_summary], feed_dict={
            x: curr_x_test_batch, y: curr_y_test_batch
        })
        test_writer.add_summary(summary=summary, global_step=count)
        print('Batch =', count, 'LearningRate =', lr, 'Loss =', loss_)

except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')

finally:
    writer = tf.summary.FileWriter("logs/", sess.graph)
    Weights_hidden_0_, biases_hidden_0_, Weights_hidden_1_, biases_hidden_1_, Weights_prediction_, \
    biases_prediction_ = sess.run([Weights_hidden_0, biases_hidden_0, Weights_hidden_1, biases_hidden_1, \
                                   Weights_prediction, biases_prediction])
    export_data = pd.DataFrame(data=list(Weights_hidden_0_) + list(biases_hidden_0_) + list(Weights_hidden_1_) + \
                                    list(biases_hidden_1_) + list(Weights_prediction_) + list(biases_prediction_))
    export_data.to_csv('results_output_3.csv')

coord.request_stop()
coord.join(threads=threads)
sess.close()
"""
