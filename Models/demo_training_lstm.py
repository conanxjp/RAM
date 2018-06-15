import tensorflow as tf
import numpy as np
import pandas as pd
import utils
import random
import sys
import argparse
import tqdm
import time

# hyperparameters
batch_iterations = 11000
batch_size = 32
full_iterations = 100
learning_rate = 0.01
reg_eta = 0.001

##########################
# dimensionalities
dim_lstm = 300
dim_aspect_embedding = 200
##########################

dim_word = 300
dim_aspect = 5
dim_sentence = 80
dim_polarity = 3

# setup utils object
isSample = False
u = utils.UTILS(batch_size, dim_sentence, dim_polarity, isSample)

# define tf placeholders
X = tf.placeholder(tf.int32, [None, dim_sentence])
y = tf.placeholder(tf.float32, [None, dim_polarity])
seqlen = tf.placeholder(tf.int32, [None])
aspects = tf.placeholder(tf.int32, [None])

# define tf variables
with tf.variable_scope('lstm_vars'):
    with tf.variable_scope('weights', reuse = tf.AUTO_REUSE):
        lstm_w = tf.get_variable(
            name = 'softmax_w',
            shape = [dim_lstm, dim_polarity],
            initializer=tf.random_uniform_initializer(-0.003, 0.003),
            regularizer = tf.contrib.layers.l2_regularizer(reg_eta)
        )
    with tf.variable_scope('biases', reuse = tf.AUTO_REUSE):
        lstm_b = tf.get_variable(
            name = 'softmax_b',
            shape = [dim_polarity],
            initializer=tf.random_uniform_initializer(-0.003, 0.003),
            regularizer = tf.contrib.layers.l2_regularizer(reg_eta)
        )


# define lstm model
def dynamic_lstm(inputs, seqlen, aspects):
#     inputs = tf.nn.dropout(inputs, keep_prob=1.0)
    with tf.name_scope('lstm_model'):
        lstm_cell = tf.contrib.rnn.LSTMCell(dim_lstm)
        outputs, state = tf.nn.dynamic_rnn(
            lstm_cell,
            inputs = inputs,
            sequence_length = seqlen,
            dtype = tf.float32,
            scope = 'lstm'
        )
        size = tf.shape(outputs)[0]
        index = tf.range(0, size) * dim_sentence + seqlen - 1 # batch_size
        output = tf.gather(tf.reshape(outputs, [-1, dim_lstm]), index)  # batch_size * n_hidden
    predict = tf.matmul(output, lstm_w) + lstm_b # batch_size x dim_polarity
    return predict

# define operations
pred = dynamic_lstm(tf.nn.embedding_lookup(u.gloveDict, X), seqlen, aspects)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = pred, labels = y), name = 'op_to_restore')
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
correct = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
init = tf.global_variables_initializer()

# tf saver
saver = tf.train.Saver()

def fullTrain():
    # full dataset training
    test_X, test_y, test_seqlen, test_aspects = u.getData('test')
    train_X, train_y, train_seqlen, train_aspects = u.getData('train')
    max_accuracy = 0.
    min_loss = 999.
    results = pd.DataFrame(columns = ['train_accuracy', 'train_loss', 'test_accuracy', 'test_loss'])
    train_a = []
    train_l = []
    test_a = []
    test_l = []
    with tf.Session() as sess:
        sess.run(init)
        for i in range(full_iterations):
            start_time = time.time()
            sess.run(optimizer, feed_dict = {X: train_X, y: train_y, seqlen: train_seqlen, aspects: train_aspects})
    #         if i > 0 and i % 4 == 0:
            loss_train, accuracy_train = sess.run([loss, accuracy], feed_dict = {X: train_X, y: train_y, seqlen: train_seqlen, aspects: train_aspects})
            print('step: %s, train loss: %s, train accuracy: %s' % (i, loss_train, accuracy_train))
            loss_test, accuracy_test = sess.run([loss, accuracy], feed_dict = {X: test_X, y: test_y, seqlen: test_seqlen, aspects: test_aspects})
            print('step: %s, test loss: %s, test accuracy: %s' % (i, loss_test, accuracy_test))
            if loss_test > loss_train and accuracy_test > max_accuracy and min_loss > loss_test:
                saver.save(sess, '../saved_model/lstm_full_train_best')
                min_loss = loss_test
                max_accuracy = accuracy_test

            end_time = time.time()
            print(end_time - start_time)
            results.loc[i] = [accuracy_train, loss_train, accuracy_test, loss_test]
            if i % 10 == 9:
                saver.save(sess, '../saved_model/lstm_full_train', global_step = i + 1)

        results.to_csv('../saved_model/results_%s_lstm.csv' % (dim_lstm), index = False)


def batchTrain():
    # batch training
    test_X, test_y, test_seqlen, test_aspects = u.getData('test')
    max_accuracy = 0.
    min_loss = 999.
    with tf.Session() as sess:
        sess.run(init)
        for i in range(batch_iterations):
            batch_X, batch_y, batch_seqlen, batch_aspects = u.nextBatch(batch_size)
            sess.run(optimizer, feed_dict = {X: batch_X, y: batch_y, seqlen: batch_seqlen, aspects: batch_aspects})
            if i > 0 and (i % 10) == 9:
                loss_train, accuracy_train = sess.run([loss, accuracy], feed_dict = {X: batch_X, y: batch_y, seqlen: batch_seqlen, aspects: batch_aspects})
                print('step: %s, train loss: %s, train accuracy: %s' % (i, loss_train, accuracy_train))
                loss_test, accuracy_test = sess.run([loss, accuracy], feed_dict = {X: test_X, y: test_y, seqlen: test_seqlen, aspects: test_aspects})
                print('step: %s, test loss: %s, test accuracy: %s' % (i, loss_test, accuracy_test))
                if loss_test > loss_train and accuracy_test > max_accuracy and min_loss > loss_test:
                    saver.save(sess, '../saved_model/lstm_batch_train_best')
                    min_loss = loss_test
                    max_accuracy = accuracy_test
                if i % 10 == 9:
                    saver.save(sess, '../saved_model/lstm_batch_train', global_step = i + 1)

if __name__ == '__main__':
    argv = sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=int, choices=[0, 1], default=0)   # 0 for full train, 1 for batch train

    args, _ = parser.parse_known_args(argv)
    runDict = {0: fullTrain, 1: batchTrain}

    # start training
    runDict[args.run]()

