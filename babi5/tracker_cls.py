# coding:utf-8

'''
  This script is for policy_babi5 (dialog state tracking).
  Created on Aug 22, 2017
  Author: qihu@mobvoi.com
'''

import os
import sys

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix

import tool.data_loader as dl


# Define some parameters for model setup, training and data loading
class Parameters():
    test_on = 0  # Whether to test[1] or train[0]
    data_opt = 0  # The options of data: for training, 0 for starting from scratch, 1 for continue
    # for testing, 0 for dev set, 1 for test set

    batch_size = 200
    test_size = 18398  # size of test set
    train_size = 18340
    dev_size = 18437
    grad_clip = 1
    epoch_num = 200
    learning_rate = 0.1
    dropout = 0.5
    l2_reg = 0.0001
    save_step = 40
    check_step = 10

    layer_num = 4
    embed_size = 20
    vocab_size = 139  # number of words used in vocabulary
    max_len = 160  # Max length (number of words) of a dialog
    state_size = 40
    fc_size = 20

    cuisine_size = 10
    location_size = 10
    number_size = 4
    price_size = 3

    data_dir = '/home/qihu/PycharmProjects/e2e-dm_babi/babi5/data'
    tmp_dir = '/home/qihu/PycharmProjects/e2e-dm_babi/babi5/tmp/'
    vocab_path = os.path.join(data_dir, 'all_vocab.txt')
    kb_path = os.path.join(data_dir, 'dialog-babi-kb-all.txt')
    kb_dir = os.path.join(data_dir, 'kb_value')
    template_path = os.path.join(data_dir, 'template', 'sys_resp.txt')
    # Train/Dev/Test path
    trn_usr_path = os.path.join(data_dir, 'tracker_usr_trn.txt')
    trn_sys_path = os.path.join(data_dir, 'tracker_sys_trn.txt')
    trn_label_path = os.path.join(data_dir, 'tracker_label_trn.txt')
    dev_usr_path = os.path.join(data_dir, 'tracker_usr_dev.txt')
    dev_sys_path = os.path.join(data_dir, 'tracker_sys_dev.txt')
    dev_label_path = os.path.join(data_dir, 'tracker_label_dev.txt')
    tst_usr_path = os.path.join(data_dir, 'tracker_usr_tst.txt')
    tst_sys_path = os.path.join(data_dir, 'tracker_sys_tst.txt')
    tst_label_path = os.path.join(data_dir, 'tracker_label_tst.txt')


# Define the Data class for data preparing
class Data(object):
    def __init__(self, params):
        self.batch_size = params.batch_size

        self.word2id, self.id2word = dl.read_word2id(params.vocab_path, params.vocab_size)
        # print self.word2id
        self.names, self.values, self.val2attr, self.entities = dl.read_kb_value(params.kb_path)

        self.trn_dialog, self.trn_dialog_vect = dl.read_tracker_dialog(params.trn_usr_path, params.trn_sys_path,
                                                                       self.word2id, params.max_len)
        self.dev_dialog, self.dev_dialog_vect = dl.read_tracker_dialog(params.dev_usr_path, params.dev_sys_path,
                                                                       self.word2id, params.max_len)
        self.tst_dialog, self.tst_dialog_vect = dl.read_tracker_dialog(params.tst_usr_path, params.tst_sys_path,
                                                                       self.word2id, params.max_len)
        self.trn_label = dl.read_tracker_label(params.trn_label_path, params.kb_dir)
        self.dev_label = dl.read_tracker_label(params.dev_label_path, params.kb_dir)
        self.tst_label = dl.read_tracker_label(params.tst_label_path, params.kb_dir)

        self.num_train = len(self.trn_label['cuisine'])
        self.num_dev = len(self.dev_label['cuisine'])
        self.num_test = len(self.tst_label['cuisine'])
        print '\tNumber of samples: train: %d, dev: %d, test: %d' % (self.num_train, self.num_dev, self.num_test)
        self._pointer = 0

    # Conversion: word-id or id-word
    def convert(self, input, relation):
        return relation[input]

    # Switch to the next batch
    def next_batch(self):
        self._pointer += self.batch_size
        if self._pointer + self.batch_size > self.num_train:
            self._pointer = 0

    # Get the train data of current batch
    def get_train_batch(self):
        start = self._pointer
        end = self._pointer + self.batch_size
        dialog = self.trn_dialog_vect[start:end]
        label_cuisine = self.trn_label['cuisine'][start:end]
        label_location = self.trn_label['location'][start:end]
        label_number = self.trn_label['number'][start:end]
        label_price = self.trn_label['price'][start:end]
        label = {'cuisine': label_cuisine,
                 'location': label_location,
                 'number': label_number,
                 'price': label_price}
        self.next_batch()
        return dialog, label

    def get_dev(self):
        return self.dev_dialog_vect, self.dev_label

    def get_tst(self):
        return self.tst_dialog_vect, self.tst_label


# Define the Seq2Seq model for dialogue system
class Tracker(object):
    def __init__(self, params):
        if test_on and data_opt:
            params.batch_size = params.dev_size
        elif test_on:
            params.batch_size = params.tst_size
        self.dropout_keep = tf.placeholder_with_default(tf.constant(1.0), shape=None, name='dropout_keep')
        self.lr = tf.placeholder_with_default(tf.constant(0.01), shape=None, name='learning_rate')
        self.x_dialog = tf.placeholder(tf.int32, shape=(None, params.max_len), name='x_dialog')
        self.y_cuisine = tf.placeholder(tf.int32, shape=(None, params.cuisine_size), name='y_cuisine')
        self.y_location = tf.placeholder(tf.int32, shape=(None, params.location_size), name='y_location')
        self.y_number = tf.placeholder(tf.int32, shape=(None, params.number_size), name='y_number')
        self.y_price = tf.placeholder(tf.int32, shape=(None, params.price_size), name='y_price')
        # Word embedding
        x_embedding = tf.get_variable(name='x_embedding', shape=[params.vocab_size+1, params.embed_size])
        x_embedded = tf.nn.embedding_lookup(x_embedding, self.x_dialog, name='x_word_embedded')
        # Extend x_api to concat with y_word_embedded

        def single_cell(state_size):  # define the cell of LSTM
            return tf.contrib.rnn.BasicLSTMCell(state_size)

        # Encoder
        self.encoder_multi_cell = tf.contrib.rnn.MultiRNNCell(
            [single_cell(params.state_size) for _ in range(params.layer_num)])  # multi-layer
        self.encoder_initial_state = self.encoder_multi_cell.zero_state(
            params.batch_size, tf.float32)  # init state of LSTM
        self.encoder_outputs, self.encoder_last_state = tf.nn.dynamic_rnn(self.encoder_multi_cell,
                                                                          x_embedded,
                                                                          initial_state=self.encoder_initial_state,
                                                                          scope='encoder')
        # Use encoder_last_state as feature
        feature = self.encoder_last_state[0][1]  # Use state h [1] (c [0]) as the feature
        self.drop = tf.nn.dropout(feature, self.dropout_keep)
        # Fully connected layer
        fc_shape = [params.state_size, params.fc_size]
        W_fc = tf.Variable(tf.truncated_normal(fc_shape, stddev=0.1), name='W_fc')
        b_fc = tf.Variable(tf.constant(0.0, shape=[params.fc_size]), name='b_fc')
        l2_loss = tf.nn.l2_loss(W_fc) + tf.nn.l2_loss(b_fc)
        self.fc = tf.nn.xw_plus_b(self.drop, W_fc, b_fc, name='fc1')
        self.fc1 = tf.nn.relu(self.fc, name='fc')
        # Softmax - act
        cuisine_shape = [params.fc_size, params.cuisine_size]
        W_cuisine = tf.Variable(tf.truncated_normal(cuisine_shape, stddev=0.1), name='W_cuisine')
        b_cuisine = tf.Variable(tf.constant(0.0, shape=[params.cuisine_size]), name='b_cuisine')
        location_shape = [params.fc_size, params.location_size]
        W_location = tf.Variable(tf.truncated_normal(location_shape, stddev=0.1), name='W_location')
        b_location = tf.Variable(tf.constant(0.0, shape=[params.location_size]), name='b_location')
        number_shape = [params.fc_size, params.number_size]
        W_number = tf.Variable(tf.truncated_normal(number_shape, stddev=0.1), name='W_number')
        b_number = tf.Variable(tf.constant(0.0, shape=[params.number_size]), name='b_number')
        price_shape = [params.fc_size, params.price_size]
        W_price = tf.Variable(tf.truncated_normal(price_shape, stddev=0.1), name='W_price')
        b_price = tf.Variable(tf.constant(0.0, shape=[params.price_size]), name='b_price')
        # Score & Sigmoid
        self.cuisine_score = tf.nn.xw_plus_b(self.fc1, W_cuisine, b_cuisine, name='cuisine_score')
        self.cuisine_prob = tf.nn.softmax(self.cuisine_score, name='cuisine_prob')
        self.location_score = tf.nn.xw_plus_b(self.fc1, W_location, b_location, name='location_score')
        self.location_prob = tf.nn.softmax(self.location_score, name='location_prob')
        self.number_score = tf.nn.xw_plus_b(self.fc1, W_number, b_number, name='number_score')
        self.number_prob = tf.nn.softmax(self.number_score, name='number_prob')
        self.price_score = tf.nn.xw_plus_b(self.fc1, W_price, b_price, name='price_score')
        self.price_prob = tf.nn.softmax(self.price_score, name='price_prob')

        cuisine_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.cuisine_score, labels=self.y_cuisine)
        location_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.location_score, labels=self.y_location)
        number_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.number_score, labels=self.y_number)
        pricez_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.price_score, labels=self.y_price)
        self.loss = tf.reduce_mean(cuisine_loss) + tf.reduce_mean(location_loss) + \
                    tf.reduce_mean(number_loss) + tf.reduce_mean(pricez_loss) + params.l2_reg * l2_loss
        print self.loss
        self.train_step = tf.train.AdamOptimizer(params.learning_rate).minimize(self.loss)
        # print 'Network set up'


# train the LSTM model
def train(data, model, params):
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    if params.data_opt:
        ckpt = tf.train.latest_checkpoint(params.tmp_dir)
        dl.optimistic_restore(sess, ckpt)
    saver = tf.train.Saver()
    dev_dialog, dev_label = data.get_dev()
    dev_feed_dict = {
        model.x_dialog: dev_dialog[:params.batch_size],
        model.y_cuisine: dev_label['cuisine'][:params.batch_size],
        model.y_location: dev_label['location'][:params.batch_size],
        model.y_number: dev_label['number'][:params.batch_size],
        model.y_price: dev_label['price'][:params.batch_size],
        model.dropout_keep: 1.0
    }
    print data.num_train
    max_iter = params.epoch_num * data.num_train / params.batch_size
    for i in range(max_iter):
        x_dialog, y_label = data.get_train_batch()
        train_feed_dict = {
            model.x_dialog: x_dialog,
            model.y_cuisine: y_label['cuisine'],
            model.y_location: y_label['location'],
            model.y_number: y_label['number'],
            model.y_price: y_label['price'],
            model.dropout_keep: 0.5
        }
        # print 'Size of Train Batch: %d, %d, %d' %(len(x_word), len(x_api), len(y_act))
        train_loss, _ = sess.run([model.loss, model.train_step], feed_dict=train_feed_dict)
        if i % params.check_step == 0:
            dev_loss = sess.run(model.loss, feed_dict=dev_feed_dict)
            print('Step: %d/%d, train_loss: %.5f, dev_loss: %.5f'
                  % (i, max_iter, train_loss, dev_loss))
        if i % params.save_step == 0:
            saver.save(sess, os.path.join(params.tmp_dir, 'e2e_model.ckpt'), global_step=i)
    sess.close()


def test(data, model, params):
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.latest_checkpoint(params.tmp_dir)
    print ckpt
    dl.optimistic_restore(sess, ckpt)
    f_true = open(os.path.join(params.tmp_dir, 'test_result_true.txt'), 'w')
    f_false = open(os.path.join(params.tmp_dir, 'test_result_false.txt'), 'w')
    f_cm = open(os.path.join(params.tmp_dir, 'test_confusion_matrix.txt'), 'w')
    error = 0
    if params.data_opt:
        x_api = data.get_api_vector(data.dev_api_num)
        x_word = data.dev_input_id
        y_act = data.get_act_vector(data.dev_label)
        label = data.dev_label
        usr_utc_list = data.dev_usr
        sys_utc_list = data.dev_sys
        api_list = data.dev_api_num
    else:
        x_api = data.get_api_vector(data.test_api_num)
        x_word = data.test_input_id
        y_act = data.get_act_vector(data.test_label)
        label = data.test_label
        usr_utc_list = data.test_usr
        sys_utc_list = data.test_sys
        api_list = data.test_api_num

    feed_dict = {
        model.x_word: x_word,
        model.x_api: x_api,
        model.y_act: y_act,
        model.dropout_keep: 1.0
    }
    print 'Size of %s Set: %d, %d, %d' % (['Test', 'Dev'][data_opt], len(x_word), len(x_api), len(y_act))
    loss, prob = sess.run([model.loss, model.prob], feed_dict=feed_dict)
    usr_utc_list = dl.flatten_2D(usr_utc_list)
    sys_utc_list = dl.flatten_2D(sys_utc_list)
    pred_list = np.zeros(len(label))
    true_list = np.asarray(label)
    pred_type_list = []
    true_type_list = []
    for i in range(len(label)):
        p = prob[i]
        act_id = np.argmax(p)
        string = '%s %s\t#%s\t#%s\t%d\n' % (data.id2act[act_id], data.id2act[label[i]], usr_utc_list[i], sys_utc_list[i], api_list[i])
        if act_id == label[i]:
            f_true.write(string)
        else:
            error += 1
            f_false.write(string)
        if data.id2act[act_id] not in pred_type_list:
            pred_type_list.append(data.id2act[act_id])
        if data.id2act[label[i]]not in true_type_list:
            true_type_list.append(data.id2act[label[i]])
        pred_list[i] = act_id
    print 'Error rate: %.3f' % (error*1.0/len(label))
    print 'Loss: %.3f' % loss
    cm = confusion_matrix(true_list, pred_list)

    # Save the confusion matrix
    f_cm.write('\t\t\t')
    for i in range(2, params.act_size+1):
        f_cm.write('%5d\t' % i)
    f_cm.write('\n')
    for i in range(params.act_size-1):
        f_cm.write('%10s\t' % data.id2act[i][:10])
        for j in range(params.act_size-1):
            f_cm.write('%5d\t' % cm[i, j])
        f_cm.write('\n')

    f_true.close()
    f_false.close()
    f_cm.close()
    sess.close()


# Main function for arguments reading
def main(test_on, data_opt=0):
    params = Parameters()
    params.test_on = test_on
    params.data_opt = data_opt
    model = Tracker(params)
    print 'Loading data ...'
    data = Data(params)
    if test_on:
        print('Testing')
        test(data, model, params)
    else:
        print('Training')
        train(data, model, params)


if __name__ == '__main__':
    msg = """
    Usage:
    Training: python e2e.py 0 0
    Testing: python e2e.py 1 0
    """
    if len(sys.argv) == 3:
        test_on = int(sys.argv[1])
        data_opt = int(sys.argv[2])
        main(test_on, data_opt)
    else:
        print(msg)
        sys.exit(1)
