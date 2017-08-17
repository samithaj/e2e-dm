# coding:utf-8

'''
  This script is for policyz_babi5 (dialog act) prediction.
  Created on Aug 16, 2017
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

    batch_size = 4000
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
    state_size = 80
    fc_size = 80

    turn_num = 1  # number of history turns
    vocab_size = 112  # number of words used in vocabulary
    utc_length = 20  # Max length of sentence, a sentence longer than max_length will be truncated
    act_size = 16

    data_dir = '/home/qihu/PycharmProjects/e2e-dm_babi/babi5/data'
    tmp_dir = '/home/qihu/PycharmProjects/e2e-dm_babi/babi5/tmp/'
    vocab_path = os.path.join(data_dir, 'all_vocab.txt')
    kb_path = os.path.join(data_dir, 'dialog-babi-kb-all.txt')
    template_path = os.path.join(data_dir, 'template', 'sys_resp.txt')
    # Train/Dev/Test path
    train_path = os.path.join(data_dir, 'dialog-babi-task5-full-dialogs-trn.txt')
    dev_path = os.path.join(data_dir, 'dialog-babi-task5-full-dialogs-dev.txt')
    test_path = os.path.join(data_dir, 'dialog-babi-task5-full-dialogs-tst.txt')


# Define the Data class for data preparing
class Data(object):
    def __init__(self, params):
        self.batch_size = params.batch_size
        self.vocab_size = params.vocab_size
        self.utc_length = params.utc_length
        self.turn_num = params.turn_num
        self.act_size = params.act_size
        self.id2act = ['you_are_welcome',
                       'request_food',
                       'api',
                       'reservation',
                       'hello',
                       'inform_address',
                       'inform_phone',
                       'request_number',
                       'on_it',
                       'any_help',
                       'find_options',
                       'update',
                       'another_option',
                       'recommend',
                       'request_area',
                       'request_price']
        self.word2id, self.id2word = dl.read_word2id(params.vocab_path, params.vocab_size)
        # print self.word2id
        self.names, self.values, self.val2attr, self.entities = dl.read_kb_value(params.kb_path)
        self.train_usr, self.train_sys, train_api = dl.read_dialog(params.train_path)
        self.dev_usr, self.dev_sys, dev_api = dl.read_dialog(params.dev_path)
        self.test_usr, self.test_sys, test_api = dl.read_dialog(params.test_path)
        self.train_label = dl.get_template_label(params.train_path, params.template_path, params.kb_path)
        self.dev_label = dl.get_template_label(params.dev_path, params.template_path, params.kb_path)
        self.test_label = dl.get_template_label(params.test_path, params.template_path, params.kb_path)
        # Merge the history turns. The number of turns to be merged is decided by params.turn_num
        train_input = dl.merge_dialog(self.train_usr, self.train_sys, params.turn_num)
        dev_input = dl.merge_dialog(self.dev_usr, self.dev_sys, params.turn_num)
        test_input = dl.merge_dialog(self.test_usr, self.test_sys, params.turn_num)
        # Flatten all history of a turn into a single string
        train_input = dl.flatten_history(train_input)
        dev_input = dl.flatten_history(dev_input)
        test_input = dl.flatten_history(test_input)
        # Convert the strings to indexes
        train_input_id = dl.convert_2D_str2id(train_input,
                                              self.word2id,
                                              self.names,
                                              self.val2attr,
                                              params.turn_num * params.utc_length,
                                              back=True)
        dev_input_id = dl.convert_2D_str2id(dev_input,
                                            self.word2id,
                                            self.names,
                                            self.val2attr,
                                            params.turn_num * params.utc_length,
                                            back=True)
        test_input_id = dl.convert_2D_str2id(test_input,
                                             self.word2id,
                                             self.names,
                                             self.val2attr,
                                             params.turn_num * params.utc_length,
                                             back=True)
        # Get number of restaurant in api_call result
        train_api_number = dl.get_api_number(train_api, train_input)
        dev_api_number = dl.get_api_number(dev_api, dev_input)
        test_api_number = dl.get_api_number(test_api, test_input)

        # Flatten the 2D list to 1D (Merge all dialogs into a single list)
        self.train_input_id = dl.flatten_2D(train_input_id)
        self.dev_input_id = dl.flatten_2D(dev_input_id)
        self.test_input_id = dl.flatten_2D(test_input_id)

        self.train_api_num = dl.flatten_2D(train_api_number)
        self.dev_api_num = dl.flatten_2D(dev_api_number)
        self.test_api_num = dl.flatten_2D(test_api_number)

        self.num_train = len(self.train_input_id)
        self.num_dev = len(self.dev_input_id)
        self.num_test = len(self.test_input_id)

        # m = 4
        # print self.train_usr[0]
        # print self.train_sys[0]
        # print self.train_input_id[m]
        # print self.train_label[m]
        # print self.train_api_num[m]

        print '\tNumber of turns: train: %d, dev: %d, test: %d' % (self.num_train, self.num_dev, self.num_test)
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
        word_id = self.train_input_id[start:end]
        api_num = self.get_api_vector(self.train_api_num[start:end])
        label_id = self.get_act_vector(self.train_label[start:end])
        self.next_batch()
        return word_id, api_num, label_id

    def get_api_vector(self, api_num):
        api_list = []
        num = len(api_num)
        for i in range(num):
            # print self.train_output_id
            api_number = np.zeros(3)
            if api_num[i] > 1:
                api_number[2] = 1
            elif api_num[i] == 1:
                api_number[1] = 1
            elif api_num[i] == 0:
                api_number[0] = 1
            api_list.append(api_number)
        return api_list

    def get_act_vector(self, act_id):
        act_list = []
        num = len(act_id)
        for i in range(num):
            # print self.train_output_id
            act_vect = np.zeros(self.act_size)
            act_vect[act_id[i]] = 1
            act_list.append(act_vect)
        return act_list


# Define the Seq2Seq model for dialogue system
class Policy(object):
    def __init__(self, params):
        # Input variable
        if params.test_on == 1:  # Test
            if params.data_opt:
                params.batch_size = params.dev_size  # On test set
            else:
                params.batch_size = params.test_size  # On dev set
        # print 'Batch_size: %d' % params.batch_size
        self.dropout_keep = tf.placeholder_with_default(tf.constant(1.0), shape=None, name='dropout_keep')
        self.lr = tf.placeholder_with_default(tf.constant(0.01), shape=None, name='learning_rate')
        self.x_word = tf.placeholder(tf.int32, shape=(None, params.turn_num * params.utc_length), name='x_word')
        self.x_api = tf.placeholder(tf.float32, shape=(None, 3), name='x_api')
        self.y_act = tf.placeholder(tf.int32, shape=(None, params.act_size), name='y_word')
        # Word embedding
        x_embedding = tf.get_variable(name='x_embedding', shape=[params.vocab_size+1, params.embed_size])
        x_word_embedded = tf.nn.embedding_lookup(x_embedding, self.x_word, name='x_word_embedded')
        # Extend x_api to concat with y_word_embedded

        def single_cell(state_size):  # define the cell of LSTM
            return tf.contrib.rnn.BasicLSTMCell(state_size)

        # Encoder
        self.encoder_multi_cell = tf.contrib.rnn.MultiRNNCell(
            [single_cell(params.state_size) for _ in range(params.layer_num)])  # multi-layer
        self.encoder_initial_state = self.encoder_multi_cell.zero_state(
            params.batch_size, tf.float32)  # init state of LSTM
        self.encoder_outputs, self.encoder_last_state = tf.nn.dynamic_rnn(self.encoder_multi_cell,
                                                                          x_word_embedded,
                                                                          initial_state=self.encoder_initial_state,
                                                                          scope='encoder')
        # Use encoder_last_state as feature (not as initial_state)
        feature = self.encoder_last_state[0][1]  # Use state h [1] (c [0]) as the feature
        feature = tf.concat([feature, self.x_api], 1)
        self.drop = tf.nn.dropout(feature, self.dropout_keep)
        # Fully connected layer
        fc_shape = [params.state_size+3, params.fc_size]
        W_fc = tf.Variable(tf.truncated_normal(fc_shape, stddev=0.1), name='W_fc')
        b_fc = tf.Variable(tf.constant(0.0, shape=[params.fc_size]), name='b_fc')
        l2_loss = tf.nn.l2_loss(W_fc) + tf.nn.l2_loss(b_fc)
        self.fc = tf.nn.xw_plus_b(self.drop, W_fc, b_fc, name='fc1')
        self.fc1 = tf.nn.relu(self.fc, name='fc')
        # Softmax - act
        act_shape = [params.fc_size, params.act_size]
        W_act = tf.Variable(tf.truncated_normal(act_shape, stddev=0.1), name='W_act')
        b_act = tf.Variable(tf.constant(0.0, shape=[params.act_size]), name='b_act')
        # Score & Sigmoid
        self.score = tf.nn.xw_plus_b(self.fc1, W_act, b_act, name='score')
        self.prob = tf.nn.softmax(self.score, name='prob')
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.score, labels=self.y_act)
        self.loss = tf.reduce_mean(loss) + params.l2_reg * l2_loss
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
    dev_api = data.get_api_vector(data.dev_api_num)
    dev_act = data.get_act_vector(data.dev_label)
    dev_feed_dict = {
        model.x_word: data.dev_input_id[:params.batch_size],
        model.x_api: dev_api[:params.batch_size],
        model.y_act: dev_act[:params.batch_size],
        model.dropout_keep: 1.0
    }

    max_iter = params.epoch_num * data.num_train / params.batch_size
    for i in range(max_iter):
        x_word, x_api, y_act = data.get_train_batch()
        train_feed_dict = {
            model.x_word: x_word,
            model.x_api: x_api,
            model.y_act: y_act,
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


def test_on_trainset(data, model, params):
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.latest_checkpoint(params.tmp_dir)
    print ckpt
    dl.optimistic_restore(sess, ckpt)
    f_true = open(os.path.join(params.tmp_dir, 'train_result_true.txt'), 'w')
    f_false = open(os.path.join(params.tmp_dir, 'train_result_false.txt'), 'w')
    f_cm = open(os.path.join(params.tmp_dir, 'train_confusion_matrix.txt'), 'w')
    error = 0
    train_api = data.get_api_vector(data.train_api_num)
    train_act = data.get_act_vector(data.train_label)
    train_feed_dict = {
        model.x_word: data.train_input_id,
        model.x_api: train_api,
        model.y_act: train_act,
        model.dropout_keep: 1.0
    }
    print 'Size of Train Set:', len(data.train_input_id), len(train_api), len(train_act)
    prob = sess.run(model.prob, feed_dict=train_feed_dict)
    train_usr = dl.flatten_2D(data.train_usr)
    train_sys = dl.flatten_2D(data.train_sys)
    pred_list = np.zeros(params.train_size)
    true_list = np.asarray(data.train_label)
    pred_type_list = []
    true_type_list = []
    for i in range(params.train_size):
        p = prob[i]
        act_id = np.argmax(p)
        string = '%s %s\t#%s\t#%s\t%d\n' % (
        data.id2act[act_id], data.id2act[data.train_label[i]], train_usr[i], train_sys[i], data.train_api_num[i])
        if act_id == data.train_label[i]:
            f_true.write(string)
        else:
            error += 1
            f_false.write(string)
        if data.id2act[act_id] not in pred_type_list:
            pred_type_list.append(data.id2act[act_id])
        if data.id2act[data.train_label[i]] not in true_type_list:
            true_type_list.append(data.id2act[data.train_label[i]])
        pred_list[i] = act_id
        # print error
    # print 'Pred type: ', pred_type_list
    # print 'True type: ', true_type_list
    # print pred_list
    print 'Error rate: %.3f' % (error * 1.0 / params.train_size)
    cm = confusion_matrix(true_list, pred_list)

    # Save the confusion matrix
    f_cm.write('\t\t\t')
    for i in range(2, params.act_size + 1):
        f_cm.write('%5d\t' % i)
    f_cm.write('\n')
    for i in range(params.act_size):
        f_cm.write('%10s\t' % data.id2act[i][:10])
        for j in range(params.act_size):
            f_cm.write('%5d\t' % cm[i, j])
        f_cm.write('\n')

    f_true.close()
    f_false.close()
    f_cm.close()
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
    model = Policy(params)
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
