import tensorflow as tf
import numpy as np
import pandas as pd
import utils
import random
import sys
import argparse

# hyperparameters
batch_iterations = 500
batch_size = 27
full_iterations = 300
learning_rate = 0.001
reg_eta = 0.001

# dimensionalities
dim_lstm = 300
dim_word = 300
dim_aspect = 5
dim_aspect_embedding = 300
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
with tf.variable_scope('aspect_embedding_vars', reuse = tf.AUTO_REUSE):
    fw_va = tf.get_variable(
        name = 'aspect_matrix_forward_Va',
        shape = [dim_aspect, dim_aspect_embedding],
        initializer = tf.random_uniform_initializer(-0.003, 0.003),
        regularizer = tf.contrib.layers.l2_regularizer(reg_eta)
    )
    bk_va = tf.get_variable(
        name = 'aspect_matrix_backward_Va',
        shape = [dim_aspect, dim_aspect_embedding],
        initializer = tf.random_uniform_initializer(-0.003, 0.003),
        regularizer = tf.contrib.layers.l2_regularizer(reg_eta)
    )
    wv = tf.get_variable(
        name = 'aspect_Wv',
        shape = [dim_aspect_embedding * 2, dim_aspect_embedding * 2],
        initializer = tf.random_uniform_initializer(-0.003, 0.003),
        regularizer = tf.contrib.layers.l2_regularizer(reg_eta)
    )
with tf.variable_scope('attention_vars', reuse = tf.AUTO_REUSE):
    wh = tf.get_variable(
        name = 'M_tanh_Wh',
        shape = [dim_lstm * 2, dim_lstm * 2],
        initializer = tf.random_uniform_initializer(-0.003, 0.003),
        regularizer = tf.contrib.layers.l2_regularizer(reg_eta)
    )
    w = tf.get_variable(
        name = 'alpha_softmax_W',
        shape = [(dim_lstm + dim_aspect_embedding) * 2, 1],
        initializer = tf.random_uniform_initializer(-0.003, 0.003),
        regularizer = tf.contrib.layers.l2_regularizer(reg_eta)
    )
    wp = tf.get_variable(
        name = 'hstar_tanh_Wp',
        shape = [dim_lstm * 2, dim_lstm * 2],
        initializer = tf.random_uniform_initializer(-0.003, 0.003),
        regularizer = tf.contrib.layers.l2_regularizer(reg_eta)
    )
    wx = tf.get_variable(
        name = 'hstar_tanh_Wx',
        shape = [dim_lstm * 2, dim_lstm * 2],
        initializer = tf.random_uniform_initializer(-0.003, 0.003),
        regularizer = tf.contrib.layers.l2_regularizer(reg_eta)
    )
with tf.variable_scope('output_softmax_vars', reuse = tf.AUTO_REUSE):
    ws = tf.get_variable(
        name = 'y_softmax_Ws',
        shape = [dim_lstm * 2, dim_polarity],
        initializer = tf.random_uniform_initializer(-0.003, 0.003),
        regularizer = tf.contrib.layers.l2_regularizer(reg_eta)
    )
    bs = tf.get_variable(
        name = 'y_softmax_Bs',
        shape = [dim_polarity],
        initializer = tf.random_uniform_initializer(-0.003, 0.003),
        regularizer = tf.contrib.layers.l2_regularizer(reg_eta)
    )


# define lstm model
def dynamic_lstm(inputs, seqlen, aspects):
#     inputs = tf.nn.dropout(inputs, keep_prob=1.0)
    with tf.name_scope('lstm_model'):
        # slice the corresponding vai from va
        fw_vai = tf.gather(fw_va, aspects) # batch_size x dim_aspect_embedding
        bk_vai = tf.gather(bk_va, aspects) # batch_size x dim_aspect_embedding
#         # concatenate vai to inputs
#         vai_en = [vai for i in range(dim_sentence)]
#         vai_en = tf.stack(vai_en, axis = 1) # batch_size x dim_sentence x dim_aspect_embedding
#         inputs = tf.concat([inputs, vai_en], 2)
        forward_lstm_cell = tf.contrib.rnn.LSTMCell(dim_lstm)
        backward_lstm_cell = tf.contrib.rnn.LSTMCell(dim_lstm)
        H, states = tf.nn.bidirectional_dynamic_rnn(
            forward_lstm_cell,
            backward_lstm_cell,
            inputs = inputs,
            sequence_length = seqlen,
            dtype = tf.float32,
            scope = 'bilstm'
        )
        fw, bk = H
        bk = tf.reverse_sequence(bk, tf.cast(seqlen, tf.int64), seq_dim=1)
        H = tf.concat(H, 2)
        size = tf.shape(H)[0]
        wv_vai = tf.matmul(tf.concat([fw_vai, bk_vai], 1), wv) # batch_size x (dim_aspect_embedding * 2)
        # stacking Wv x Va along sentence length
        wv_vai = [wv_vai for i in range(dim_sentence)]
        wv_vai_en = tf.stack(wv_vai, axis = 1) # batch_size x dim_sentence x (dim_aspect_embedding * 2)
        wv_vai_en = tf.reshape(wv_vai_en, [-1, dim_aspect_embedding * 2]) # (batch_size * dim_sentence) x (dim_aspect_embedding * 2)
        H_1 = tf.reshape(H, [-1, dim_lstm * 2]) # (batch_size * dim_sentence) x (dim_lstm * 2)
        wh_H = tf.matmul(H_1, wh) # (batch_size * dim_sentence) x (dim_lstm * 2)
        # concatenate wh_H and wv_va_En for inputting to tanh
        wh_H_wv_vai_en = tf.concat([wh_H, wv_vai_en], 1) # (batch_size * dim_sentence) x [(dim_lstm + dim_aspect_embedding) * 2]
        M = tf.tanh(wh_H_wv_vai_en) # (batch_size * dim_sentence) x [(dim_lstm + dim_aspect_embedding) * 2]
        alpha = tf.nn.softmax(tf.matmul(M, w)) # (batch_size * dim_sentence)
        alpha = tf.reshape(alpha, [-1, 1, dim_sentence]) # batch_size x 1 x dim_sentence
        index = tf.range(0, size) * dim_sentence + seqlen - 1 # batch_size
        hn = tf.gather(tf.reshape(H, [-1, dim_lstm * 2]), index)  # batch_size x (dim_lstm * 2)
        r = tf.reshape(tf.matmul(alpha, H), [-1, dim_lstm * 2]) # batch_size x (dim_lstm * 2)
        h_star = tf.tanh(tf.matmul(r, wp) + tf.matmul(hn, wx)) # batch_size x (dim_lstm * 2)
        predict = tf.matmul(h_star, ws) + bs # batch x dim_polarity
    return predict

# define operations
pred = dynamic_lstm(tf.nn.embedding_lookup(u.gloveDict, X), seqlen, aspects)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = pred, labels = y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
correct = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
init = tf.global_variables_initializer()

def fullTrain():
    # full dataset training
    test_X, test_y, test_seqlen, test_aspects = u.getData('test')
    train_X, train_y, train_seqlen, train_aspects = u.getData('train')
    with tf.Session() as sess:
        sess.run(init)
        for i in range(full_iterations):
            sess.run(optimizer, feed_dict = {X: train_X, y: train_y, seqlen: train_seqlen, aspects: train_aspects})
    #         if i > 0 and i % 4 == 0:
            loss_train, accuracy_train = sess.run([loss, accuracy], feed_dict = {X: train_X, y: train_y, seqlen: train_seqlen, aspects: train_aspects})
            print('step: %s, train loss: %s, train accuracy: %s' % (i, loss_train, accuracy_train))
            loss_test, accuracy_test = sess.run([loss, accuracy], feed_dict = {X: test_X, y: test_y, seqlen: test_seqlen, aspects: test_aspects})
            print('step: %s, test loss: %s, test accuracy: %s' % (i, loss_test, accuracy_test))

def batchTrain():
    # batch training
    test_X, test_y, test_seqlen, test_aspects = u.getData('test')
    with tf.Session() as sess:
        sess.run(init)
        for i in range(batch_iterations):
            batch_X, batch_y, batch_seqlen, batch_aspects = u.nextBatch(batch_size)
            sess.run(optimizer, feed_dict = {X: batch_X, y: batch_y, seqlen: batch_seqlen, aspects: batch_aspects})
            if i > 0 and i % 4 == 0:
                loss_train, accuracy_train = sess.run([loss, accuracy], feed_dict = {X: batch_X, y: batch_y, seqlen: batch_seqlen, aspects: batch_aspects})
                print('step: %s, train loss: %s, train accuracy: %s' % (i, loss_train, accuracy_train))
                loss_test, accuracy_test = sess.run([loss, accuracy], feed_dict = {X: test_X, y: test_y, seqlen: test_seqlen, aspects: test_aspects})
                print('step: %s, test loss: %s, test accuracy: %s' % (i, loss_test, accuracy_test))

if __name__ == '__main__':
    argv = sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=int, choices=[0, 1], default=0)   # 0 for full train, 1 for batch train

    args, _ = parser.parse_known_args(argv)
    runDict = {0: fullTrain, 1: batchTrain}

    # start training
    runDict[args.run]()
