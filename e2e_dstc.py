# coding:utf-8

'''
  This script is for e2e_dm modeling (on DSTC2 dataset).
  Created on Aug 8, 2017
  Author: qihu@mobvoi.com
'''

import os
import sys

import numpy as np
import tensorflow as tf
import tool.data_loader as dl


# Define some parameters for model setup, training and data loading
class Parameters():
    batch_size = 100
    grad_clip = 1
    epoch_num = 100
    learning_rate = 0.1
    dropout = 0.5
    save_step = 100
    check_step = 10

    turn_num = 3  # number of history turns
    vocab_size = 770  # number of words used in vocabulary
    utc_length = 20  # Max length of sentence, a sentence longer than max_length will be truncated
    layer_num = 1

    data_dir = 'data'
    tmp_dir = 'tmp'
    vocab_path = os.path.join(data_dir, 'all_vocab.txt')
    kb_path = os.path.join(data_dir, 'dialog-babi-task6-dstc2-kb.txt')
    # Train/Dev/Test path
    train_path = os.path.join(data_dir, 'dialog-babi-task6-dstc2-trn.txt')
    dev_path = os.path.join(data_dir, 'dialog-babi-task6-dstc2-dev.txt')
    test_path = os.path.join(data_dir, 'dialog-babi-task6-dstc2-tst.txt')


# Define the Data class for data preparing
class Data(object):
    def __init__(self, params):
        self.batch_size = params.batch_size
        self.vocab_size = params.vocab_size
        self.utc_length = params.utc_length

        self.word2id, self.id2word = dl.read_word2id(params.word2id_path, params.vocab_size)
        self.names, self.values, self.val2attr, self.entities = dl.read_kb_value(params.kb_path)
        self.train_usr, self.train_sys, train_api = dl.read_dialog(params.train_path)
        self.dev_usr, self.dev_sys, dev_api = dl.read_dialog(params.dev_path)
        self.test_usr, self.test_sys, test_api = dl.read_dialog(params.test_path)
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
                                              params.turn_num*params.utc_length)
        dev_input_id = dl.convert_2D_str2id(dev_input,
                                              self.word2id,
                                              self.names,
                                              self.val2attr,
                                              params.turn_num * params.utc_length)
        test_input_id = dl.convert_2D_str2id(test_input,
                                              self.word2id,
                                              self.names,
                                              self.val2attr,
                                              params.turn_num * params.utc_length)
        train_output_id = dl.convert_2D_str2id(self.train_sys,
                                              self.word2id,
                                              self.names,
                                              self.val2attr,
                                              params.turn_num * params.utc_length)
        dev_output_id = dl.convert_2D_str2id(self.dev_sys,
                                            self.word2id,
                                            self.names,
                                            self.val2attr,
                                            params.turn_num * params.utc_length)
        test_output_id = dl.convert_2D_str2id(self.test_sys,
                                             self.word2id,
                                             self.names,
                                             self.val2attr,
                                             params.turn_num * params.utc_length)
        # Get number of restaurant in api_call result
        train_api_number = dl.get_api_number(train_api, train_input)
        dev_api_number = dl.get_api_number(dev_api, dev_input)
        test_api_number = dl.get_api_number(test_api, test_input)

        # Flatten the 2D list to 1D (Merge all dialogs into a single list)
        self.train_input_id = dl.flatten_2D(train_input_id)
        self.dev_input_id = dl.flatten_2D(dev_input_id)
        self.test_input_id = dl.flatten_2D(test_input_id)

        self.train_output_id = dl.flatten_2D(train_output_id)
        self.dev_output_id = dl.flatten_2D(dev_output_id)
        self.test_output_id = dl.flatten_2D(test_output_id)

        self.train_api_num = dl.flatten_2D(train_api_number)
        self.dev_api_num = dl.flatten_2D(dev_api_number)
        self.test_api_num = dl.flatten_2D(test_api_number)

        self._pointer = 0

    # Conversion: word-id or id-word
    def convert(self, input, relation):
        return relation[input]

    # Switch to the next batch
    def next_batch(self):
        self._pointer += self.batch_size
        if self._pointer + self.batch_size > self.data_num:
            self._pointer = 0

    # Get the data of current batch
    def get_batch(self):
        start = self._pointer
        end = self._pointer+self.batch_size

        usr_list = self.train_input[start:end]
        api_number_list = self.train_api_num[start:end]
        sys_in_list = []
        sys_out_list = []
        for i in range(self.batch_size):
            sys_in = np.zeros(self.utc_length)
            sys_out = np.zeros(self.utc_length)
            sys_in[:-1] = self.train_output[start + i][:-1]
            sys_out[:-1] = self.train_output[start + i][1:]
            sys_in_list.append(sys_in)
            sys_out_list.append(sys_out)
        return usr_list, api_number_list, sys_in_list, sys_out_list


# Define the Seq2Seq model for dialogue system
class Seq2Seq(object):
    def __init__(self, params):
        # Input variable
        self.dropout_keep = tf.placeholder_with_default(tf.constant(1.0), shape=None)
        self.lr = tf.placeholder_with_default(tf.constant(0.01), shape=None)
        self.x_word = tf.placeholder(tf.int32, shape=(None, params.turn_num * params.utc_length), name='x_word')
        self.x_api = tf.placeholder(tf.float32, shape=(None, 1), name='x_api')
        self.y_word_in = tf.placeholder(tf.int32, shape=(None, params.utc_length), name='y_word')
        self.y_word_out = tf.placeholder(tf.int32, shape=(None, params.utc_length), name='y_word')
        # Word embedding
        x_embedding = tf.get_variable(name='x_embedding', shape=[params.vocab_size, params.embed_size])
        x_word_embedded = tf.nn.embedding_lookup(x_embedding, self.x_word)
        y_embedding = tf.get_variable(name='y_embedding', shape=[params.vocab_size, params.embed_size])
        y_word_embedded = tf.nn.embedding_lookup(y_embedding, self.y_word_in)

        def single_cell(state_size):  # define the cell of LSTM
            return tf.contrib.rnn.BasicLSTMCell(state_size)

        # Encoder
        self.encoder_multi_cell = tf.contrib.rnn.MultiRNNCell([single_cell(params.state_size) for _ in range(params.layer_num)])  # multi-layer
        self.encoder_initial_state = self.encoder_multi_cell.zero_state(params.batch_size, tf.float32)  # init state of LSTM
        outputs, self.encoder_last_state = tf.nn.dynamic_rnn(self.encodermulti_cell,
                                                             x_word_embedded,
                                                             initial_state=self.encoder_initial_state)

        self.feature = tf.concat([self.encoder_last_state, self.x_api], 1)
        # Decoder

        self.decoder_multi_cell = tf.contrib.rnn.MultiRNNCell([single_cell(params.state_size+1) for _ in range(params.layer_num)])  # multi-layer
        self.decoder_initial_state = self.feature
        outputs, self.decoder_last_state = tf.nn.dynamic_rnn(self.decoder_multi_cell,
                                                             y_word_embedded,
                                                             initial_state=self.decoder_initial_state)

        self.w = tf.get_variable("softmax_w", [params.state_size, params.sys_vocab_size])  # weights for output
        self.b = tf.get_variable("softmax_b", [params.sys_vocab_size])

        # Loss
        output = tf.reshape(outputs, [-1, params.state_size])
        self.logits = tf.matmul(output, self.w) + self.b
        self.probs = tf.nn.softmax(self.logits)
        targets = tf.reshape(self.y_word_out, [-1])
        weights = tf.ones_like(targets, dtype=tf.float32)
        # print self.logits, targets, weights
        loss = tf.contrib.legacy_seq2seq.sequence_loss([self.logits], [targets], [weights])
        self.cost = tf.reduce_sum(loss) / params.batch_size
        optimizer = tf.train.AdamOptimizer(self.lr)
        tvars = tf.trainable_variables()
        grads = tf.gradients(self.cost, tvars)
        grads, _ = tf.clip_by_global_norm(grads, params.grad_clip)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))


# train the LSTM model
def train(data, model, params):
    print 'Training ...'
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    max_iter = params.epoch_num*params.data_num/params.batch_size
    for i in range(1, max_iter+1):
        x_word, x_vd, x_state, x_act, x_slot, y_word_in, y_word_out = data.get_batch()
        # print y_word_in[0]
        # print y_word_out[0]
        feed_dict = {model.x_word: x_word,
                     model.x_vd: x_vd,
                     model.x_state: x_state,
                     model.y_word_in: y_word_in,
                     model.y_word_out: y_word_out,
                     model.lr: params.learning_rate,
                     model.dropout_keep: params.dropout
                    }
        train_loss, _ = sess.run([model.cost, model.train_op], feed_dict=feed_dict)
        if i % params.check_step == 0:
            print('Step: %d/%d, training_loss: %.4f' % (i, max_iter, train_loss))
        if i % params.save_step == 0:
            saver.save(sess, os.path.join(params.tmp_dir, 'e2e_model.ckpt'), global_step=i)
        data.next_batch()  # Switch to next batch
    sess.close()


def test(data, model, params):
    print 'Testing ...'


# Main function for arguments reading
def main(infer):
    params = Parameters()
    params.infer = infer
    data = Data(params)
    model = Seq2Seq(params)
    if infer:
        print('Testing')
        test(data, model, params)
    else:
        print('Training')
        train(data, model, params)


if __name__ == '__main__':
    msg = """
    Usage:
    Training: python e2e.py 0
    Testing: python e2e.py 1
    Testing with beam search: python e2e.py 2
    Cross Validation: python e2e.py 3
    """
    if len(sys.argv) == 2:
        infer = int(sys.argv[-1])
        main(infer)
    else:
        print(msg)
        sys.exit(1)
