import tensorflow as tf
import numpy as np
import pandas as pd
import config as cf
from preprocess import (processData, prepareData)
import time

print('preparing data for model ...')
trainData, testData, validData, sampleData = prepareData()

def get_batch_index(length, batch_size, is_shuffle=True):
    index = list(range(length))
    if is_shuffle:
        np.random.shuffle(index)
    return index

# hyperparameters
batch_iterations = 500
batch_size = 32
full_iterations = 50
learning_rate = 0.005
reg_eta = 0.001
dim_lstm = 300
num_AL = 3


# dimensionalities
dim_word = 300
dim_sentence = 80
dim_polarity = 3

train_word_weights = np.load(cf.FOLDER + 'word_positions_train.npy')
test_word_weights = np.load(cf.FOLDER + 'word_positions_test.npy')
valid_word_weights = np.load(cf.FOLDER + 'word_positions_valid.npy')
sample_word_weights = np.load(cf.FOLDER + 'word_positions_sample.npy')
glove = np.load(cf.FOLDER + 'glove.npy')
timestamp = str(int(time.time()))


temp, valid_aspects_encoding = validData
valid_X, valid_y, valid_seqlen = temp
valid_X = np.array(valid_X)
valid_seqlen = np.array(valid_seqlen)
temp, sample_aspects_encoding = sampleData
sample_X, sample_y, sample_seqlen = temp
temp, train_aspects_encoding = trainData
train_X, train_y, train_seqlen = temp
train_X = np.array(train_X)
train_seqlen = np.array(train_seqlen)
temp, test_aspects_encoding = testData
test_X, test_y, test_seqlen = temp


X = tf.placeholder(tf.int32, [None, dim_sentence])
aspects = tf.placeholder(tf.float32, [None, dim_word])
seqlen = tf.placeholder(tf.int32, None)
sentence_locs = tf.placeholder(tf.float32, [None, dim_sentence])
y = tf.placeholder(tf.int32, [None, dim_polarity])

with tf.variable_scope('attention', reuse = tf.AUTO_REUSE):
    Wal = tf.get_variable(
        name='W_al',
        shape=[num_AL, 1, dim_lstm * 3 + dim_word + 1],
        # initializer=tf.random_uniform_initializer(-0.003, 0.003),
        initializer=tf.contrib.layers.xavier_initializer(),
        regularizer=tf.contrib.layers.l2_regularizer(reg_eta)
    )
    Bal = tf.get_variable(
        name='B_al',
        shape=[num_AL, 1, dim_sentence],
        # initializer=tf.random_uniform_initializer(-0.003, 0.003),
        initializer=tf.zeros_initializer(),
        regularizer=tf.contrib.layers.l2_regularizer(reg_eta)
    )

with tf.variable_scope('gru', reuse = tf.AUTO_REUSE):
    Wr = tf.get_variable(
        name='W_r',
        shape=[dim_lstm, dim_lstm * 2 + 1],
        # initializer=tf.random_uniform_initializer(-0.003, 0.003),
        initializer=tf.orthogonal_initializer(),
        regularizer=tf.contrib.layers.l2_regularizer(reg_eta)
    )
    Wz = tf.get_variable(
        name='W_z',
        shape=[dim_lstm, dim_lstm * 2 + 1],
        # initializer=tf.random_uniform_initializer(-0.003, 0.003),
        initializer=tf.orthogonal_initializer(),
        regularizer=tf.contrib.layers.l2_regularizer(reg_eta)
    )
    Wg = tf.get_variable(
        name='W_g',
        shape=[dim_lstm, dim_lstm],
        # initializer=tf.random_uniform_initializer(-0.003, 0.003),
        initializer=tf.orthogonal_initializer(),
        regularizer=tf.contrib.layers.l2_regularizer(reg_eta)
    )
    Wx = tf.get_variable(
        name='W_x',
        shape=[dim_lstm, dim_lstm * 2 + 1],
        # initializer=tf.random_uniform_initializer(-0.003, 0.003),
        initializer=tf.orthogonal_initializer(),
        regularizer=tf.contrib.layers.l2_regularizer(reg_eta)
    )
    Ur = tf.get_variable(
        name='U_r',
        shape=[dim_lstm, dim_lstm],
        # initializer=tf.random_uniform_initializer(-0.003, 0.003),
        initializer=tf.orthogonal_initializer(),
        regularizer=tf.contrib.layers.l2_regularizer(reg_eta)
    )
    Uz = tf.get_variable(
        name='U_z',
        shape=[dim_lstm, dim_lstm],
        # initializer=tf.random_uniform_initializer(-0.003, 0.003),
        initializer=tf.orthogonal_initializer(),
        regularizer=tf.contrib.layers.l2_regularizer(reg_eta)
    )

with tf.variable_scope('softmax', reuse = tf.AUTO_REUSE):
    Ws = tf.get_variable(
        name='W_s',
        shape=[dim_lstm, dim_polarity],
        # initializer=tf.random_uniform_initializer(-0.003, 0.003),
        initializer=tf.contrib.layers.xavier_initializer(),
        regularizer=tf.contrib.layers.l2_regularizer(reg_eta)
    )
    Bs = tf.get_variable(
        name='B_s',
        shape=[dim_polarity],
        # initializer=tf.random_uniform_initializer(-0.003, 0.003),
        initializer=tf.zeros_initializer(),
        regularizer=tf.contrib.layers.l2_regularizer(reg_eta)
    )

    def dynamic_lstm(inputs, seqlen, weights, aspects):
        inputs = tf.cast(tf.nn.dropout(inputs, keep_prob=0.5), tf.float32)
        with tf.name_scope('bilstm_model'):
            forward_lstm_cell = tf.contrib.rnn.LSTMCell(dim_lstm)
            backward_lstm_cell = tf.contrib.rnn.LSTMCell(dim_lstm)
            Mstar, states = tf.nn.bidirectional_dynamic_rnn(
                forward_lstm_cell,
                backward_lstm_cell,
    #             tf.unstack(tf.transpose(inputs, perm=[1, 0, 2])),
                inputs = inputs,
                sequence_length = seqlen,
                dtype = tf.float32,
                scope = 'bilstm'
            )
            fw, bk = Mstar
    #         bk = tf.reverse_sequence(bk, tf.cast(seqlen, tf.int64), seq_dim=1)
            Mstar = tf.concat([fw, bk], 2)
            # Mstar is batch_size x [(dim_sentence x dim_lstm), (dim_sentence x dim_lstm)]
    #         Mstar = tf.reshape(tf.concat(Mstar, 1), [-1, dim_sentence, dim_lstm * 2]) # batch_size x dim_sentence x (dim_lstm * 2)
            batch_size = tf.shape(Mstar)[0]
            u = 1 - weights # batch_size x dim_sentence
            weights = [weights for i in range(dim_lstm * 2)] # stack weights vectors
            weights = tf.stack(weights, 1) # batch_size x (dim_lstm * 2) x dim_sentence
            weights = tf.reshape(weights, [-1, dim_sentence, dim_lstm * 2])
            M = tf.multiply(Mstar, weights) # batch_size x dim_sentence x (dim_lstm * 2)
            u = tf.reshape(u, [-1, dim_sentence])
            M = tf.reshape(M, [-1, dim_sentence * (dim_lstm * 2)])
            M = tf.concat([M, u], 1) # batch_size x (dim_sentence * (dim_lstm * 2 + 1))
            M = tf.reshape(M, [-1, dim_sentence, dim_lstm * 2 + 1]) # batch_size x dim_sentence x (dim_lstm * 2 + 1)

            # recurrently update attention on memory (M)
            e = tf.zeros([batch_size, dim_lstm])
    #         i_AL = tf.random_uniform([batch_size, dim_lstm * 2 + 1], -0.5, 0.5)
    #         i_AL = tf.ones([batch_size, dim_lstm * 2 + 1])
    #         e_stack = [e for i in range(dim_sentence)]
    #         e_stack = tf.stack(e_stack, 1)
    #         e_stack = tf.reshape(e_stack, [-1, dim_lstm])
    #         aspects_stack = [aspects for i in range(dim_sentence)]
    #         aspects_stack = tf.stack(aspects_stack, 1)
    #         aspects_stack = tf.reshape(aspects_stack, [-1, dim_word])
    #         e_aspects_concat = tf.concat([e_stack, aspects_stack], 1)
    #         M_unfold = tf.reshape(M, [-1, dim_lstm * 2 + 1])
    #         all_concat = tf.concat([M_unfold, e_aspects_concat], 1) # (batch_size * dim_sentence) x (dim_lstm * 2 + 1 + dim_lstm + dim_word)
    # #         all_concat = tf.reshape(all_concat, [batch_size, dim_sentence, -1])
    #         g = tf.matmul(Wal[0], tf.transpose(all_concat)) # 1 x (batch_size * dim_sentence)
    #         alpha = tf.nn.softmax(tf.reshape(g, [-1]))
    #         alpha = tf.reshape(alpha, [-1, 1, dim_sentence]) # batch_size x 1 x dim_sentence
    #         i_AL = tf.matmul(alpha, M) # batch_size x 1 x (dim_lstm * 2 + 1)
    #         i_AL = tf.reshape(i_AL, [-1, dim_lstm * 2 + 1]) # batch_size x (dim_lstm * 2 + 1)
            for al in range(num_AL):


                # update i_AL
                e_stack = [e for i in range(dim_sentence)]
                e_stack = tf.stack(e_stack, 1)
                e_stack = tf.reshape(e_stack, [-1, dim_lstm])
                aspects_stack = [aspects for i in range(dim_sentence)]
                aspects_stack = tf.stack(aspects_stack, 1)
                aspects_stack = tf.reshape(aspects_stack, [-1, dim_word])
                e_aspects_concat = tf.concat([e_stack, aspects_stack], 1)
                M_unfold = tf.reshape(M, [-1, dim_lstm * 2 + 1])
                all_concat = tf.concat([M_unfold, e_aspects_concat], 1) # (batch_size * dim_sentence) x (dim_lstm * 2 + 1 + dim_lstm + dim_word)
                g = tf.matmul(Wal[al], tf.transpose(all_concat)) # 1 x (batch_size * dim_sentence)
                alpha = tf.nn.softmax(tf.reshape(g, [-1]))
                alpha = tf.reshape(alpha, [-1, 1, dim_sentence]) # batch_size x 1 x dim_sentence
                i_AL = tf.matmul(alpha, M) # batch_size x 1 x (dim_lstm * 2 + 1)
                i_AL = tf.reshape(i_AL, [-1, dim_lstm * 2 + 1]) # batch_size x (dim_lstm * 2 + 1)
                print(i_AL)
                # last step in gru: update e
                # gru
                r = tf.nn.sigmoid(tf.matmul(i_AL, tf.transpose(Wr)) + tf.matmul(e, Ur)) # batch_size x dim_lstm
                z = tf.nn.sigmoid(tf.matmul(i_AL, tf.transpose(Wz)) + tf.matmul(e, Uz)) # batch_size x dim_lstm
                e_temp = tf.nn.tanh(tf.matmul(i_AL, tf.transpose(Wx)) + tf.matmul(tf.multiply(r, e), Wg)) # batch_size x dim_lstm
                e = tf.multiply((1 - z), e) + tf.multiply(z, e_temp) # batch_size x dim_lstm

            predict = tf.matmul(e, Ws) + Bs # dim_polarity
        return predict

predict = dynamic_lstm(tf.nn.embedding_lookup(glove, X), seqlen, sentence_locs, aspects)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = predict, labels = y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
correct = tf.equal(tf.argmax(predict, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(full_iterations):
        index = get_batch_index(len(train_X), batch_size, True)
        l = 0.
        a = 0.
        c = 0
        for j in range(int(len(index) / batch_size) + (1 if len(index) % batch_size else 0)):
            _, batch_loss, batch_accuracy = sess.run([optimizer, loss, accuracy], feed_dict = {X: train_X[index[j * batch_size : (j + 1) * batch_size]], y: train_y[index[j * batch_size : (j + 1) * batch_size]], seqlen: train_seqlen[index[j * batch_size : (j + 1) * batch_size]], sentence_locs: train_word_weights[index[j * batch_size : (j + 1) * batch_size]], aspects: train_aspects_encoding[index[j * batch_size : (j + 1) * batch_size]]})
            l += batch_loss
            a += batch_accuracy
            c += 1
        print(print('epoch: %s, train loss: %s, train accuracy: %s' % (i, l/c, a/c)))
        loss_valid, accuracy_valid = sess.run([loss, accuracy], feed_dict = {X: valid_X, y: valid_y, seqlen: valid_seqlen, sentence_locs: valid_word_weights, aspects: valid_aspects_encoding})
        print('epoch: %s, valid loss: %s, valid accuracy: %s' % (i, loss_valid, accuracy_valid))
