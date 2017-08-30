import os
import argparse
import tensorflow as tf
import numpy as np

from tool import data_loader as dl


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
    vocab_size = 143  # number of words used in vocabulary
    gen_len = 2  # length of output [start, end]
    max_len = 160  # Max length (number of words) of a dialog
    state_size = 40
    fc_size = 20
    end_of_sequence_id = 0
    init_min_val = -0.1
    init_max_val = 0.1
    num_glimpse = 1

    cuisine_size = 10
    location_size = 10
    number_size = 4
    price_size = 3

    slots = ['cuisine', 'location', 'number', 'price']

    data_dir = '/home/qihu/PycharmProjects/e2e-dm_babi/babi5/data'
    tmp_dir = '/home/qihu/PycharmProjects/e2e-dm_babi/babi5/tmp/ptr'
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
        self.trn_label_id = dl.read_tracker_label(params.trn_label_path, self.word2id, params.slots)
        self.dev_label_id = dl.read_tracker_label(params.dev_label_path, self.word2id, params.slots)
        self.tst_label_id = dl.read_tracker_label(params.tst_label_path, self.word2id, params.slots)

        self.trn_label_pos = dl.get_tracker_label_pos(self.trn_dialog_vect, params.trn_label_path, self.word2id, params.slots)
        self.dev_label_pos = dl.get_tracker_label_pos(self.dev_dialog_vect, params.dev_label_path, self.word2id, params.slots)
        self.tst_label_pos = dl.get_tracker_label_pos(self.tst_dialog_vect, params.tst_label_path, self.word2id, params.slots)

        self.num_train = len(self.trn_label_id['cuisine'])
        self.num_dev = len(self.dev_label_id['cuisine'])
        self.num_test = len(self.tst_label_id['cuisine'])
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
        dialogs = self.trn_dialog_vect[start:end]*4
        label_pos = []
        label_id = []
        for s in params.slots:
            label_id.extend(self.trn_label_id[s][start:end])
            label_pos.extend(self.trn_label_pos[s][start:end])
        self.next_batch()
        return dialogs, label_id, label_pos

    def get_dev(self):
        dialogs = self.dev_dialog_vect * 4
        label_pos = []
        label_id = []
        for s in params.slots:
            label_id.extend(self.dev_label_id[s])
            label_pos.extend(self.dev_label_pos[s])
        return dialogs, label_id, label_pos

    def get_tst(self):
        dialogs = self.tst_dialog_vect * 4
        label_pos = []
        label_id = []
        for s in params.slots:
            label_id.extend(self.tst_label_id[s])
            label_pos.extend(self.tst_label_pos[s])
        return dialogs, label_id, label_pos


class Pointer(object):
    def __init__(self, params):
        if params.is_train:
            self.batch_size = 4*params.batch_size
        else:
            self.batch_size = params.batch_size
        # dropout keep probability and learning rate
        self.dropout_keep = tf.placeholder_with_default(tf.constant(1.0), shape=None, name='dropout_keep')
        self.lr = tf.placeholder_with_default(tf.constant(0.01), shape=None, name='learning_rate')
        # Input placeholder
        self.x_dialog = tf.placeholder(tf.int32, shape=(self.batch_size, params.max_len), name='input_dialog')
        self.y_id_in = tf.placeholder(tf.int32, shape=(self.batch_size, params.gen_len), name='input_prev_id')
        self.y_pos_out = tf.placeholder(tf.int32, shape=(self.batch_size, params.gen_len), name='output_position')

        enc_outputs, enc_last_state = self.encoder(self.x_dialog, params)
        initializer = tf.random_uniform_initializer(params.init_min_val, params.init_max_val)
        dec_outputs, dec_last_state, dec_last_context_state = self.decoder_from_ptr(self.y_id_in,
                                                                                    enc_outputs,
                                                                                    enc_last_state,
                                                                                    initializer=initializer,
                                                                                    max_length=params.gen_len)

        w = tf.get_variable("softmax_w", [params.max_len, params.max_len])  # weights for output
        b = tf.get_variable("softmax_b", [params.max_len])
        output = tf.reshape(dec_outputs, [-1, params.max_len])
        logits = tf.matmul(output, w) + b
        self.probs = tf.nn.softmax(logits)
        targets = tf.reshape(self.y_pos_out, [-1])
        weights = tf.ones_like(targets, dtype=tf.float32)

        self.loss = tf.contrib.legacy_seq2seq.sequence_loss([logits], [targets], [weights])
        # self.loss = tf.reduce_sum(loss) / params.batch_size
        optimizer = tf.train.AdamOptimizer(self.lr)
        tvars = tf.trainable_variables()
        grads = tf.gradients(self.loss, tvars)
        grads, _ = tf.clip_by_global_norm(grads, params.grad_clip)
        self.train_step = optimizer.apply_gradients(zip(grads, tvars))

    # Single LSTM cell
    def single_cell(self, state_size):  # define the cell of LSTM
        return tf.contrib.rnn.BasicLSTMCell(state_size)

    # Encoder
    def encoder(self, x_dialog, params):
        # Embedding for dialog
        x_dialog_embedding = tf.get_variable(shape=[1+params.vocab_size, params.embed_size], name='x_dialog_embedding')
        x_dialog_embedded = tf.nn.embedding_lookup(x_dialog_embedding, x_dialog, name='x_dialog_embedded')
        encoder_multi_cell = tf.contrib.rnn.MultiRNNCell(
            [self.single_cell(params.state_size) for _ in range(params.layer_num)])  # multi-layer
        encoder_initial_state = encoder_multi_cell.zero_state(self.batch_size, tf.float32)  # init state of LSTM
        encoder_outputs, encoder_last_state = tf.nn.dynamic_rnn(encoder_multi_cell,
                                                                x_dialog_embedded,
                                                                initial_state=encoder_initial_state,
                                                                scope='encoder')
        return encoder_outputs, encoder_last_state

    # Decoder coypied from pointer-network-tensorflow
    def decoder_from_ptr(self, inputs, enc_outputs, enc_final_states, initializer=None, max_length=None):
        decoder_multi_cell = tf.contrib.rnn.MultiRNNCell(
            [self.single_cell(params.state_size) for _ in range(params.layer_num)])
        y_id_embedding = tf.get_variable(shape=[params.vocab_size, params.embed_size], name='y_pos_embedding')
        y_id_embedded = tf.nn.embedding_lookup(y_id_embedding, inputs, name='y_id_embedded')

        def attention(ref, query, with_softmax, scope="attention"):
            with tf.variable_scope(scope):
                W_ref = tf.get_variable(
                    "W_ref", [1, params.state_size, params.state_size], initializer=initializer)
                W_q = tf.get_variable(
                    "W_q", [params.state_size, params.state_size], initializer=initializer)
                v = tf.get_variable(
                    "v", [params.state_size], initializer=initializer)

                encoded_ref = tf.nn.conv1d(ref, W_ref, 1, "VALID", name="encoded_ref")
                encoded_query = tf.expand_dims(tf.matmul(query, W_q, name="encoded_query"), 1)
                scores = tf.reduce_sum(v * tf.tanh(encoded_ref + encoded_query), [-1])

                if with_softmax:
                    return tf.nn.softmax(scores)
                else:
                    return scores

        def index_matrix_to_pairs(index_matrix):
            # [[3,1,2], [2,3,1]] -> [[[0, 3], [1, 1], [2, 2]],
            #                        [[0, 2], [1, 3], [2, 1]]]
            replicated_first_indices = tf.range(tf.shape(index_matrix)[0])
            rank = len(index_matrix.get_shape())
            if rank == 2:
                replicated_first_indices = tf.tile(
                    tf.expand_dims(replicated_first_indices, dim=1),
                    [1, tf.shape(index_matrix)[1]])
            return tf.stack([replicated_first_indices, index_matrix], axis=rank)

        def glimpse(ref, query, scope="glimpse"):
            p = attention(ref, query, with_softmax=True, scope=scope)
            alignments = tf.expand_dims(p, 2)
            return tf.reduce_sum(alignments * ref, [1])

        def output_fn(ref, query, num_glimpse):
            if query is None:
                return tf.zeros([max_length], tf.float32)  # only used for shape inference
            else:
                for idx in range(num_glimpse):
                    query = glimpse(ref, query, "glimpse_{}".format(idx))
                return attention(ref, query, with_softmax=False, scope="attention")

        def input_fn(sampled_idx):
            return tf.stop_gradient(
                tf.gather_nd(enc_outputs, index_matrix_to_pairs(sampled_idx)))

        decoder_fn = tf.contrib.seq2seq.simple_decoder_fn_train(enc_final_states)

        # if params.is_train:
        #     decoder_fn = tf.contrib.seq2seq.simple_decoder_fn_train(enc_final_states)
        # else:
        #     # decoder_fn = tf.contrib.seq2seq.simple_decoder_fn_inference(output_fn, enc_final_states, y_id_embedding,
        #     #                                                             0, params.max_len, params.gen_len, params.max_len)
        #
        #     maximum_length = tf.convert_to_tensor(params.gen_len, tf.int32)
        #
        #     def decoder_fn(time, cell_state, cell_input, cell_output, context_state):
        #         cell_output = output_fn(enc_outputs, cell_output, params.num_glimpse)
        #         if cell_state is None:
        #             cell_state = enc_final_states
        #             next_input = cell_input
        #             done = tf.zeros([self.batch_size, ], dtype=tf.bool)
        #         else:
        #             sampled_idx = tf.cast(tf.argmax(cell_output, 1), tf.int32)
        #             next_input = input_fn(sampled_idx)
        #             done = tf.equal(sampled_idx, params.end_of_sequence_id)
        #
        #         done = tf.cond(tf.greater(time, maximum_length),
        #                        lambda: tf.ones([self.batch_size, ], dtype=tf.bool),
        #                        lambda: done)
        #         return done, cell_state, next_input, cell_output, context_state

        outputs, final_state, final_context_state = \
            tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_multi_cell, decoder_fn, y_id_embedded, params.gen_len)
        if True:  # params.is_train:
            transposed_outputs = tf.transpose(outputs, [1, 0, 2])
            fn = lambda x: output_fn(enc_outputs, x, params.num_glimpse)
            outputs = tf.transpose(tf.map_fn(fn, transposed_outputs), [1, 0, 2])
        return outputs, final_state, final_context_state

    # Decoder coypied from pointer-generator-master (Not used yet)
    def decoder_from_ptrgtr(decoder_inputs, initial_state, encoder_states, enc_padding_mask, cell,
                          initial_state_attention=False, pointer_gen=True, use_coverage=False, prev_coverage=None):
        """
        Args:
          decoder_inputs: A list of 2D Tensors [batch_size x input_size].
          initial_state: 2D Tensor [batch_size x cell.state_size].
          encoder_states: 3D Tensor [batch_size x attn_length x attn_size].
          enc_padding_mask: 2D Tensor [batch_size x attn_length] containing 1s and 0s; indicates which of the encoder locations are padding (0) or a real token (1).
          cell: rnn_cell.RNNCell defining the cell function and size.
          initial_state_attention:
            Note that this attention decoder passes each decoder input through a linear layer with the previous step's context vector to get a modified version of the input. If initial_state_attention is False, on the first decoder step the "previous context vector" is just a zero vector. If initial_state_attention is True, we use initial_state to (re)calculate the previous step's context vector. We set this to False for train/eval mode (because we call attention_decoder once for all decoder steps) and True for decode mode (because we call attention_decoder once for each decoder step).
          pointer_gen: boolean. If True, calculate the generation probability p_gen for each decoder step.
          use_coverage: boolean. If True, use coverage mechanism.
          prev_coverage:
            If not None, a tensor with shape (batch_size, attn_length). The previous step's coverage vector. This is only not None in decode mode when using coverage.

        Returns:
          outputs: A list of the same length as decoder_inputs of 2D Tensors of
            shape [batch_size x cell.output_size]. The output vectors.
          state: The final state of the decoder. A tensor shape [batch_size x cell.state_size].
          attn_dists: A list containing tensors of shape (batch_size,attn_length).
            The attention distributions for each decoder step.
          p_gens: List of scalars. The values of p_gen for each decoder step. Empty list if pointer_gen=False.
          coverage: Coverage vector on the last step computed. None if use_coverage=False.
        """
        with variable_scope.variable_scope("attention_decoder") as scope:
            batch_size = encoder_states.get_shape()[
                0].value  # if this line fails, it's because the batch size isn't defined
            attn_size = encoder_states.get_shape()[
                2].value  # if this line fails, it's because the attention length isn't defined

            # Reshape encoder_states (need to insert a dim)
            encoder_states = tf.expand_dims(encoder_states, axis=2)  # now is shape (batch_size, attn_len, 1, attn_size)

            # To calculate attention, we calculate
            #   v^T tanh(W_h h_i + W_s s_t + b_attn)
            # where h_i is an encoder state, and s_t a decoder state.
            # attn_vec_size is the length of the vectors v, b_attn, (W_h h_i) and (W_s s_t).
            # We set it to be equal to the size of the encoder states.
            attention_vec_size = attn_size

            # Get the weight matrix W_h and apply it to each encoder state to get (W_h h_i), the encoder features
            W_h = variable_scope.get_variable("W_h", [1, 1, attn_size, attention_vec_size])
            encoder_features = nn_ops.conv2d(encoder_states, W_h, [1, 1, 1, 1],
                                             "SAME")  # shape (batch_size,attn_length,1,attention_vec_size)

            # Get the weight vectors v and w_c (w_c is for coverage)
            v = variable_scope.get_variable("v", [attention_vec_size])
            if use_coverage:
                with variable_scope.variable_scope("coverage"):
                    w_c = variable_scope.get_variable("w_c", [1, 1, 1, attention_vec_size])

            if prev_coverage is not None:  # for beam search mode with coverage
                # reshape from (batch_size, attn_length) to (batch_size, attn_len, 1, 1)
                prev_coverage = tf.expand_dims(tf.expand_dims(prev_coverage, 2), 3)

            def attention(decoder_state, coverage=None):
                """Calculate the context vector and attention distribution from the decoder state.

                Args:
                  decoder_state: state of the decoder
                  coverage: Optional. Previous timestep's coverage vector, shape (batch_size, attn_len, 1, 1).

                Returns:
                  context_vector: weighted sum of encoder_states
                  attn_dist: attention distribution
                  coverage: new coverage vector. shape (batch_size, attn_len, 1, 1)
                """
                with variable_scope.variable_scope("Attention"):
                    # Pass the decoder state through a linear layer (this is W_s s_t + b_attn in the paper)
                    decoder_features = linear(decoder_state, attention_vec_size,
                                              True)  # shape (batch_size, attention_vec_size)
                    decoder_features = tf.expand_dims(tf.expand_dims(decoder_features, 1),
                                                      1)  # reshape to (batch_size, 1, 1, attention_vec_size)

                    def masked_attention(e):
                        """Take softmax of e then apply enc_padding_mask and re-normalize"""
                        attn_dist = nn_ops.softmax(e)  # take softmax. shape (batch_size, attn_length)
                        attn_dist *= enc_padding_mask  # apply mask
                        masked_sums = tf.reduce_sum(attn_dist, axis=1)  # shape (batch_size)
                        return attn_dist / tf.reshape(masked_sums, [-1, 1])  # re-normalize

                    if use_coverage and coverage is not None:  # non-first step of coverage
                        # Multiply coverage vector by w_c to get coverage_features.
                        coverage_features = nn_ops.conv2d(coverage, w_c, [1, 1, 1, 1],
                                                          "SAME")  # c has shape (batch_size, attn_length, 1, attention_vec_size)

                        # Calculate v^T tanh(W_h h_i + W_s s_t + w_c c_i^t + b_attn)
                        e = math_ops.reduce_sum(
                            v * math_ops.tanh(encoder_features + decoder_features + coverage_features),
                            [2, 3])  # shape (batch_size,attn_length)

                        # Calculate attention distribution
                        attn_dist = masked_attention(e)

                        # Update coverage vector
                        coverage += array_ops.reshape(attn_dist, [batch_size, -1, 1, 1])
                    else:
                        # Calculate v^T tanh(W_h h_i + W_s s_t + b_attn)
                        e = math_ops.reduce_sum(v * math_ops.tanh(encoder_features + decoder_features),
                                                [2, 3])  # calculate e

                        # Calculate attention distribution
                        attn_dist = masked_attention(e)

                        if use_coverage:  # first step of training
                            coverage = tf.expand_dims(tf.expand_dims(attn_dist, 2), 2)  # initialize coverage

                    # Calculate the context vector from attn_dist and encoder_states
                    context_vector = math_ops.reduce_sum(
                        array_ops.reshape(attn_dist, [batch_size, -1, 1, 1]) * encoder_states,
                        [1, 2])  # shape (batch_size, attn_size).
                    context_vector = array_ops.reshape(context_vector, [-1, attn_size])

                return context_vector, attn_dist, coverage

    # Decoder (AttentionWrapper(cannot get the attention scare)) (Not used yet)
    def decoder(self, inputs, enc_outputs, enc_last_state):
        y_embedding = tf.get_variable(shape=[params.max_len, params.embed_size], name='y_embedding')
        y_embedded = tf.nn.embedding_lookup(y_embedding, inputs, name='y_embeded')

        decoder_multi_cell = tf.contrib.rnn.MultiRNNCell(
            [self.single_cell(params.state_size) for _ in range(params.layer_num)])  # multi-layer

        attn_mech = tf.contrib.seq2seq.BahdanauAttention(num_units=params.state_size,  # LuongAttention
                                                         memory=enc_outputs,
                                                         name='attention_mechanic')
        attn_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_multi_cell,
                                                        attention_mechanism=attn_mech,
                                                        attention_layer_size=128,
                                                        name="attention_wrapper")
        attn_zero = attn_cell.zero_state(batch_size=params.batch_size,
                                         dtype=tf.float32)
        train_helper = tf.contrib.seq2seq.TrainingHelper(inputs=y_embedded,
                                                         sequence_length=params.gen_len,
                                                         time_major=False)
        projection_layer = Dense(params.vocab_size)
        decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=attn_cell,  # attn_cell,
            helper=train_helper,  # A Helper instance
            initial_state=attn_zero.clone(cell_state=enc_last_state),  # initial state of decoder
            output_layer=projection_layer)  # instance of tf.layers.Layer, like Dense

        # Perform dynamic decoding with decoder
        dec_outputs, dec_last_state, dec_last_context_state = tf.contrib.seq2seq.dynamic_decode(decoder=decoder,
                                                                           impute_finished=True,
                                                                           maximum_iterations=gen_length)
        return dec_outputs, dec_last_state, dec_last_context_state


def train(data, model, params):
    print 'Train...'
    sess = tf.Session()
    dev_sess = tf.Session()
    train_batch_size = params.batch_size
    params.batch_size = data.num_dev
    with tf.variable_scope('dev'):
        dev_model = Pointer(params)
    params.batch_size = train_batch_size
    sess.run(tf.global_variables_initializer())
    if params.restore:
        ckpt = tf.train.latest_checkpoint(params.tmp_dir)
        dl.optimistic_restore(sess, ckpt)
    saver = tf.train.Saver()
    dev_dialog, dev_label_id, dev_label_pos = data.get_dev()
    ##  Set size of devset to batch_size
    # dev_dialog = dev_dialog[0:params.batch_size]+\
    #              dev_dialog[data.num_dev: data.num_dev+params.batch_size]+\
    #              dev_dialog[2*data.num_dev: 2*data.num_dev+params.batch_size]+\
    #              dev_dialog[3*data.num_dev: 3*data.num_dev+params.batch_size]
    # dev_label_id = dev_label_id[0:params.batch_size] + \
    #                dev_label_id[data.num_dev: data.num_dev + params.batch_size] + \
    #                dev_label_id[2 * data.num_dev: 2 * data.num_dev + params.batch_size] + \
    #                dev_label_id[3 * data.num_dev: 3 * data.num_dev + params.batch_size]
    # dev_label_pos = dev_label_pos[0:params.batch_size] + \
    #                 dev_label_pos[data.num_dev: data.num_dev + params.batch_size] + \
    #                 dev_label_pos[2 * data.num_dev: 2 * data.num_dev + params.batch_size] + \
    #                 dev_label_pos[3 * data.num_dev: 3 * data.num_dev + params.batch_size]
    # print len(dev_dialog), len(dev_label_id), len(dev_label_pos)
    dev_feed_dict = {
        model.x_dialog: dev_dialog,
        model.y_id_in: dev_label_id,
        model.y_pos_out: dev_label_pos,
        model.dropout_keep: 1.0
    }
    max_iter = params.epoch_num * data.num_train / (4*params.batch_size)
    for i in range(max_iter):
        x_dialog, y_pos_in, y_pos_out = data.get_train_batch()
        train_feed_dict = {
            model.x_dialog: x_dialog,
            model.y_id_in: y_pos_in,
            model.y_pos_out: y_pos_out,
            model.dropout_keep: 0.5
        }
        # print 'Size of Train Batch: %d, %d, %d' %(len(x_word), len(x_api), len(y_act))
        train_loss, _ = sess.run([model.loss, model.train_step], feed_dict=train_feed_dict)
        if i % params.check_step == 0:
            ckpt = tf.train.latest_checkpoint(params.tmp_dir)
            dl.optimistic_restore(dev_sess, ckpt)
            dev_loss = sess.run(dev_model.loss, feed_dict=dev_feed_dict)
            print('Step: %d/%d, train_loss: %.5f, dev_loss: %.5f'
                  % (i, max_iter, train_loss, dev_loss))
        if i % params.save_step == 0:
            saver.save(sess, os.path.join(params.tmp_dir, 'ptr_model.ckpt'), global_step=i)
    sess.close()
    dev_sess.close()


def test(data, model, params):
    print 'test...'
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.latest_checkpoint(params.tmp_dir)
    dl.optimistic_restore(sess, ckpt)
    tst_dialog, tst_label_id, tst_label_pos = data.get_tst()
    num_data = len(tst_dialog)
    accuracy = 0
    tst_feed_dict = {
        model.x_dialog: tst_dialog,
        model.y_id_in: tst_label_id,
        model.y_pos_out: tst_label_pos,
        model.dropout_keep: 1.0
    }
    # print 'Size of Train Batch: %d, %d, %d' %(len(x_word), len(x_api), len(y_act))
    tst_loss, tst_prob = sess.run([model.loss, model.probs], feed_dict=tst_feed_dict)
    tst_prob = tst_prob.reshape([num_data, params.gen_len, params.max_len])
    f = open(os.path.join(params.tmp_dir, 'ptr_test_result.txt'), 'w')
    for i in range(num_data):
        y_pos_start = np.argmax(tst_prob[i][0])
        y_pos_end = np.argmax(tst_prob[i][1])
        pred = ''
        gt = ''
        slot = data.convert(int(tst_label_id[i][0]), data.id2word)
        for j in range(y_pos_start, y_pos_end+1):
            pred = pred + data.convert(int(tst_dialog[i][j]), data.id2word) + ' '
        for j in range(int(tst_label_pos[i][0]), int(tst_label_pos[i][1]+1)):
            gt = gt + data.convert(int(tst_dialog[i][j]), data.id2word) + ' '
        # print y_pos_start, y_pos_end, "\t", slot, gt, pred
        f.write('%s\t%d %d\t%s %s\n' % (slot, y_pos_start, y_pos_end, gt, pred))
        if y_pos_start == int(tst_label_pos[i][0]) and y_pos_end == int(tst_label_pos[i][1]):
            accuracy += 1
    accuracy = accuracy*1.0/len(tst_dialog)
    print 'Accuracy: %.3f' % accuracy
    f.close()
    sess.close()


# Test on testset one by one
def test_individual(data, model, params):
    print 'test...'
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.latest_checkpoint(params.tmp_dir)
    dl.optimistic_restore(sess, ckpt)
    tst_dialog, tst_label_id, tst_label_pos = data.get_tst()
    num_data = len(tst_dialog)
    accuracy = 0
    for i in range(num_data):
        x_dialog = tst_dialog[i]
        y_id_in = tst_label_id[i]
        y_pos_out = tst_label_pos[i]
        tst_feed_dict = {
            model.x_dialog: [x_dialog],
            model.y_id_in: [y_id_in],
            model.y_pos_out: [y_pos_out],
            model.dropout_keep: 1.0
        }
        # print 'Size of Train Batch: %d, %d, %d' %(len(x_word), len(x_api), len(y_act))
        tst_loss, tst_prob = sess.run([model.loss, model.probs], feed_dict=tst_feed_dict)
        y_pos_start = np.argmax(tst_prob[0])
        y_pos_end = np.argmax(tst_prob[1])
        pred = ''
        gt = ''
        slot = data.convert(int(y_id_in[0]), data.id2word)
        for j in range(y_pos_start, y_pos_end+1):
            pred = pred + data.convert(int(x_dialog[j]), data.id2word) + ' '
        for j in range(int(y_pos_out[0]), int(y_pos_out[1]+1)):
            gt = gt + data.convert(int(x_dialog[j]), data.id2word) + ' '
        print tst_loss, y_pos_start, y_pos_end, "\t", slot, gt, pred
        if y_pos_start == int(y_pos_out[0]) and y_pos_end == int(y_pos_out[1]):
            accuracy += 1
    accuracy = accuracy*1.0/len(tst_dialog)
    print 'Accuracy: %.3f' % accuracy
    sess.close()


def test_old(data, model, params):
    print 'Test...'
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.latest_checkpoint(params.tmp_dir)
    print ckpt
    dl.optimistic_restore(sess, ckpt)

    tst_dialog, tst_label_id, tst_label_pos = data.get_tst()

    f = open(os.path.join(params.tmp_dir, 'ptr_test_result.txt'), 'w')
    for i in range(4*data.num_test):
        gt_start = tst_label_pos[i][0]
        gt_end = tst_label_pos[i][1]
        pred_range = np.zeros(2)

        x_dialog = tst_dialog[i]
        x_label_id = np.zeros(1)
        x_label_id[0] = tst_label_id[i][0]
        for j in range(params.gen_len):
            feed_dict = {
                model.x_dialog: [x_dialog],
                model.y_id_in: [x_label_id],
                model.y_pos_out: [tst_label_pos[0][j]],
                model.dropout_keep: 1.0
            }
            probs = sess.run(model.probs, feed_dict)
            print probs
            p = probs[0]
            x_label_id = np.argmax(p)
            pred_range[j] = x_label_id

        slot = data.convert(tst_label_id[i][0], data.id2word)
        gt = ''
        for j in range(gt_start, gt_end + 1):  # get ground truth string
            gt = gt + data.convert(x_dialog[0][j], data.id2word) + ' '
        pred = ''
        for j in range(pred_range[0], pred_range[1] + 1):  # get predication string
            pred = pred + data.convert(x_dialog[0][j], data.id2word) + ' '
        answer = '%s\t%s\t%s' % (slot, gt, pred)
        show_str = ('%d\t%s\n\t%s\n' % (i + 1, x_raw_str, answer))
        print(show_str)
        f.write('%s\n' % show_str)
    sess.close()
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pointer Networks")
    parser.add_argument('-t', '--task', help='0 for train, 1 for test', action="store_true")
    parser.add_argument('-r', '--restore', help='restore training from history', action="store_true")
    args = parser.parse_args()
    params = Parameters()
    params.is_train = args.task
    params.restore = args.restore
    data = Data(params)
    if params.is_train:
        model = Pointer(params)
        train(data, model, params)
    else:
        params.batch_size = 4*data.num_test
        model = Pointer(params)
        test(data, model, params)
