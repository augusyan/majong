# -*- coding:utf-8 -*-
"""
@author : Augus Yan
@file : train_demo_v1.py
@time : 2017/10/15 11:10
@function : 
"""
from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
import tensorflow as tf
import numpy as np

def read_traindata():
    filename_queue = tf.train.string_input_producer(["file0.csv", "file1.csv"])
    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)

    # Default values, in case of empty columns. Also specifies the type of the
    # decoded result.
    record_defaults = [[1], [1], [1], [1], [1]]
    col1, col2, col3, col4, col5 = tf.decode_csv(
        value, record_defaults=record_defaults)
    features = tf.concat(0, [col1, col2, col3, col4])

    with tf.Session() as sess:
        # Start populating the filename queue.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(1200):
            # Retrieve a single instance:
            example, label = sess.run([features, col5])

        coord.request_stop()
        coord.join(threads)


# 添加层
def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


# Convert class labels from scalars to one-hot vectors.
def dense_to_one_hot(labels_dense, num_classes):

    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


# one-hot 另一个方法
"""def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels
"""
# 1.全局变量
feature_dim = 50
nb_classes = 34
nb_nerul = 80
# training_epochs = 15
# batch_size = 100
# learning_rate = 0.1

# 2.读取数据集
# x_data = np.loadtxt("D:\\datasets\\ex4x.dat").astype(np.float32)
# y_data = np.loadtxt("D:\\datasets\\ex4y.dat").astype(np.float32)
"""csvTrain = np.genfromtxt('train.csv', delimiter=",")
X = np.array(csvTrain[:, :225]) #225, 15
Y = csvTrain[:,225]

csvTest = np.genfromtxt('test.csv', delimiter=",")
testX = np.array(csvTest[:, :225])
testY = csvTest[:,225]

#reshape features for each instance in to 15*15, targets are just a single number
X = X.reshape([-1,15,15,1])
testX = testX.reshape([-1,15,15,1])

## Building convolutional network
network = input_data(shape=[None, 15, 15, 1], name='input')
"""
x_data = np.random.rand(100, 50).astype(np.float32) * 34
y_data = np.random.randint(0, 34, size=(100, 1))
y_data = dense_to_one_hot(y_data, 34)
X = tf.placeholder(tf.float32, [None, feature_dim], name='X')
Y = tf.placeholder(tf.float32, [None, nb_classes], name='Y')
# Y = tf.placeholder(tf.float32, [None])
# Y_one_hot = tf.one_hot(Y, 1)
keep_prob = tf.placeholder(tf.float32)  # dropout (keep_prob) rate  0.7 on training


# 3.weights & bias for nn layers
W_fc1 = tf.Variable(tf.random_normal([feature_dim, nb_nerul]), name='W_fc1')
b_fc1 = tf.Variable(tf.zeros([1, nb_nerul]), name='b_fc1')
hfc_1 = tf.nn.relu(tf.matmul(X, W_fc1) + b_fc1)
# hfc_1 = tf.nn.dropout(hfc_1, keep_prob=1.0)
# hfc_1 = tf.reshape(hfc_1, [-1, 40])
# W_1 = tf.truncated_normal([30, 80], stddev=0.1, name='W_1')
W_fc2 = tf.Variable(tf.random_normal([nb_nerul, nb_classes]), name='W_fc2')
b_fc2 = tf.Variable(tf.zeros([1, nb_classes]), name='b_fc2')
hfc_2 = tf.nn.softmax(tf.matmul(hfc_1, W_fc2) + b_fc2)
# hfc_2 = tf.nn.dropout(hfc_1, keep_prob=1.0)
# _hidden_layer = add_layer(X, 30, 80, activation_function=tf.nn.relu)
# prediction = add_layer(_hidden_layer, 80, 15, activation_function=None)

# 4.定义 loss 表达式
# the error between prediciton and real data
# cost = tf.reduce_mean(tf.reduce_sum(tf.square(Y - hypothesis), axis=[1]))
# define cost/loss & optimizer
# cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hfc_2, labels=Y))
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
optimizer = tf.train.AdamOptimizer(learning_rate=0.1).minimize(cost)
predicted = tf.equal(tf.argmax(hfc_2, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(predicted, dtype=tf.float32))

saver = tf.train.Saver()  # 保存模型
# 5.initialize
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    feed = {X: x_data, Y: y_data}
    # 6.train my model
    """for epoch in range(200):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            feed_dict = {X: batch_xs, Y: batch_ys}
            c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
            avg_cost += c / total_batch
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
    """
    for step in range(20001):
        sess.run(optimizer, feed_dict=feed)
        if step % 100 == 0:
            print(step, sess.run(cost, feed_dict=feed))
        # Accuracy report
    h, c, a = sess.run([hfc_2, predicted, accuracy], feed_dict=feed)
    print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)
    # print(W_fc2.eval())
    saver_path = saver.save(sess, "D:\\model.ckpt")
    # 将模型保存到save/model.ckpt文件
    print("Model saved in file:", saver_path)
print('Learning Finished!')
# Test model and check accuracy
# correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# print('Accuracy:', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))

# 测试模型

# 读取模型
"""w2_hist = tf.summary.histogram("weights2", W_2)
cost_summ = tf.summary.scalar("cost", cost)
summary = tf.summary.merge_all()
writer = tf.summary.FileWriter('.//logs')
writer.add_graph(sess.graph)
s, _ = sess.run([summary, optimizer], feed_dict=feed)
writer.add_summary(s, global_step=global_step)"""


"""
# 1.训练的数据
# Make up some real data
x_data = np.linspace(-1,1,300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

# 2.定义节点准备接收数据
# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

# 3.定义神经层：隐藏层和预测层
# add hidden layer 输入值是 xs，在隐藏层有 10 个神经元
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# add output layer 输入值是隐藏层 l1，在预测层输出 1 个结果
prediction = add_layer(l1, 10, 1, activation_function=None)

# 4.定义 loss 表达式
# the error between prediciton and real data
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                     reduction_indices=[1]))

# 5.选择 optimizer 使 loss 达到最小
# 这一行定义了用什么方式去减少 loss，学习率是 0.1
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)


# important step 对所有变量进行初始化
init = tf.global_variables_initializer()
sess = tf.Session()
# 上面定义的都没有运算，直到 sess.run 才会开始运算
sess.run(init)

# 迭代 1000 次学习，sess.run optimizer
for i in range(1000):
    # training train_step 和 loss 都是由 placeholder 定义的运算，所以这里要用 feed 传入参数
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        # to see the step improvement
        print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
        """
