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
    batch_size = 40
    grad_clip = 1
    epoch_num = 200
    learning_rate = 0.1
    dropout = 0.5
    save_step = 40
    check_step = 10

    turn_num = 3  # number of history turns
    vocab_size = 770  # number of words used in vocabulary
    utc_length = 20  # Max length of sentence, a sentence longer than max_length will be truncated
    gen_length = 20
    layer_num = 1
    embed_size = 21
    state_size = 30

    data_dir = '/home/qihu/PycharmProjects/e2e-dm_babi/data'
    tmp_dir = '/home/qihu/PycharmProjects/e2e-dm_babi/tmp/seq2seq'
    vocab_path = os.path.join(data_dir, 'all_vocab.txt')
    kb_path = os.path.join(data_dir, 'dialog-babi-task6-data-kb.txt')
    # Train/Dev/Test path
    train_path = os.path.join(data_dir, 'dialog-babi-task6-data-trn.txt')
    dev_path = os.path.join(data_dir, 'dialog-babi-task6-data-dev.txt')
    test_path = os.path.join(data_dir, 'dialog-babi-task6-data-tst.txt')


# Define the Data class for data preparing
class Data(object):
    def __init__(self, params):
        self.batch_size = params.batch_size
        self.vocab_size = params.vocab_size
        self.utc_length = params.utc_length
        self.turn_num = params.turn_num

        self.word2id, self.id2word = dl.read_word2id(params.vocab_path, params.vocab_size)
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
                                              params.utc_length)
        dev_output_id = dl.convert_2D_str2id(self.dev_sys,
                                            self.word2id,
                                            self.names,
                                            self.val2attr,
                                            params.utc_length)
        test_output_id = dl.convert_2D_str2id(self.test_sys,
                                             self.word2id,
                                             self.names,
                                             self.val2attr,
                                             params.utc_length)
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

        self.num_train = len(self.train_input_id)
        self.num_dev = len(self.dev_input_id)
        self.num_test = len(self.test_input_id)
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
        end = self._pointer+self.batch_size
        input_id = self.train_input_id[start:end]
        api_num = self.train_api_num[start:end]
        output_id = self.train_output_id[start:end]
        self.next_batch()
        return self.get_batch(input_id, api_num, output_id)

    # Get the dev data
    def get_batch(self, input_id, api_num, output_id):
        usr_list = input_id
        api_number_list = []
        sys_in_list = []
        sys_out_list = []
        # print self.train_input_id
        for i in range(self.batch_size):
            # print self.train_output_id
            api_number = np.zeros(3)
            if api_num[i] > 1:
                api_number[2] = 1
            elif api_num[i] == 1:
                api_number[1] = 1
            elif api_num[i] == 0:
                api_number[0] = 1
            api_number_list.append(api_number)
            sys_in = np.zeros(self.utc_length)
            sys_out = np.zeros(self.utc_length)
            sys_in[:-1] = output_id[i][:-1]
            sys_out[:-1] = output_id[i][1:]
            sys_in_list.append(sys_in)
            sys_out_list.append(sys_out)
        return usr_list, api_number_list, sys_in_list, sys_out_list


# Define the Seq2Seq model for dialogue system
class Seq2Seq(object):
    def __init__(self, params):
        # Input variable
        if infer == 1:
            params.batch_size = 1
            params.gen_length = 1

        self.dropout_keep = tf.placeholder_with_default(tf.constant(1.0), shape=None)
        self.lr = tf.placeholder_with_default(tf.constant(0.01), shape=None)
        # self.x_word = [tf.placeholder(tf.int32, shape=[None], name='x_word')]*params.batch_size
        # self.y_word_in = [tf.placeholder(tf.int32, shape=[None], name='y_word')]*params.batch_size
        # self.y_word_out = [tf.placeholder(tf.int32, shape=[None], name='y_word')]*params.batch_size
        # self.x_api = [tf.placeholder(tf.float32, shape=[None], name='x_api')] * params.batch_size
        self.y_word_out = tf.placeholder(tf.int32, shape=[None, params.gen_length], name='y_word')
        self.x_api = tf.placeholder(tf.float32, shape=[None, 3], name='x_api')

        # Feeds for inputs.
        self.x_word = []
        self.y_word_in = []
        for i in xrange(params.batch_size):  # Last bucket is the biggest one.
            self.x_word.append(tf.placeholder(tf.int32, shape=[None], name="encoder{0}".format(i)))
            self.y_word_in.append(tf.placeholder(tf.int32, shape=[None], name="decoder{0}".format(i)))

        outputs, states = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
            self.x_word,
            self.y_word_in,
            cell=tf.contrib.rnn.BasicLSTMCell(params.state_size),
            num_encoder_symbols=params.vocab_size,
            num_decoder_symbols=params.vocab_size,
            embedding_size=params.embed_size)

        self.w = tf.get_variable("softmax_w", [params.state_size, params.vocab_size])  # weights for output
        self.b = tf.get_variable("softmax_b", [params.vocab_size])

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
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    x_word, x_api, y_word_in, y_word_out = data.get_batch(data.dev_input_id,
                                                          data.dev_api_num,
                                                          data.dev_output_id)
    print params.batch_size
    print x_word[:params.batch_size][0].shape
    dev_feed_dict = {
        model.x_word: x_word[:params.batch_size],
        model.x_api: x_api[:params.batch_size],
        model.y_word_in: y_word_in[:params.batch_size],
        model.y_word_out: y_word_out[:params.batch_size],
        model.dropout_keep: 1.0
    }

    max_iter = params.epoch_num*data.num_train/params.batch_size
    for i in range(max_iter):
        x_word, x_api, y_word_in, y_word_out = data.get_train_batch()
        feed_dict = {
            model.x_word: x_word,
            model.x_api: x_api,
            model.y_word_in: y_word_in,
            model.y_word_out: y_word_out,
            model.dropout_keep: 0.5
        }

        train_loss, _ = sess.run([model.cost, model.train_op], feed_dict=feed_dict)
        # print tf.Print(model.encoder_outputs, [model.encoder_outputs[0][59][0]])
        # print tf.Print(model.encoder_last_state, [model.encoder_last_state[0][0]])
        if i % params.check_step == 0:
            dev_loss = sess.run(model.cost, feed_dict=dev_feed_dict)
            print('Step: %d/%d, train_loss: %.5f, dev_loss: %.5f'
                  % (i, max_iter, train_loss, dev_loss))
        if i % params.save_step == 0:
            saver.save(sess, os.path.join(params.tmp_dir, 'e2e_model.ckpt'), global_step=i)

    sess.close()


def test(data, model, params):
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.latest_checkpoint(params.tmp_dir)
    dl.optimistic_restore(sess, ckpt)
    test_x_word, test_x_api, test_y_word_in, test_y_word_out = \
        data.get_batch(data.test_input_id, data.test_api_num, data.test_output_id)

    for i in range(data.num_test):
        x_word = [test_x_word[i]]
        x_api = [test_x_api[i]]
        y_word_in = [test_y_word_in[i]]
        y_word_out = [test_y_word_out[i]]
        # Run encoder just once
        state = sess.run(model.encoder_multi_cell.zero_state(1, tf.float32))
        feed_dict = {
            model.x_word: x_word,
            model.encoder_initial_state: state,
        }
        encoder_last_state = sess.run(model.encoder_last_state, feed_dict)
        state = sess.run(model.decoder_multi_cell.zero_state(1, tf.float32))
        # Run decoder
        answer = ''
        word = '<s>'
        for j in range(params.utc_length):
            x = np.zeros([1, 1])
            x[0, 0] = data.convert(word, data.word2id)
            feed_dict = {
                model.x_api: x_api,
                model.y_word_in: x,
                model.encoder_last_state: encoder_last_state,
                model.decoder_initial_state: state,
            }
            probs, state = sess.run([model.probs, model.decoder_last_state], feed_dict)
            p = probs[0]
            word = data.convert(np.argmax(p), data.id2word)
            # if word == '</s>':
            #     break
            answer += word
        show_str = ('%d\t%s' % (i + 1, unicode(answer, encoding='utf-8')))
        print(show_str)
    sess.close()


# Main function for arguments reading
def main(infer):
    params = Parameters()
    params.infer = infer
    model = Seq2Seq(params)
    print 'Loading data ...'
    data = Data(params)
    if infer:
        print('Testing')
        test(data, model, params)
    else:
        print('Training')
        train(data, model, params)


if __name__ == '__main__':
    msg = """
    Usage:
    Training: python seq2seq.py 0
    Testing: python seq2seq.py 1
    """
    if len(sys.argv) == 2:
        infer = int(sys.argv[-1])
        main(infer)
    else:
        print(msg)
        sys.exit(1)
