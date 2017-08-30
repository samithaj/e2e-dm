import tensorflow as tf
from tensorflow.python.layers.core import Dense

# INPUT
x_word = tf.placeholder(tf.int32, shape=(None, None), name='x_word')
x_api = tf.placeholder(tf.float32, shape=(None, None), name='x_api')
y_word_in = tf.placeholder(tf.int32, shape=(None, 20), name='y_word')
# Word embedding
x_embedding = tf.get_variable(name='x_embedding', shape=[771, 20])
x_word_embedded = tf.nn.embedding_lookup(x_embedding, x_word)
y_embedding = tf.get_variable(name='y_embedding', shape=[771, 20])
y_word_embedded = tf.nn.embedding_lookup(y_embedding, y_word_in)
# Extend x_api to concat with y_word_embedded
x_api = tf.expand_dims(x_api, 1)
x_api_extend = x_api
for i in range(20 - 1):
    x_api_extend = tf.concat([x_api_extend, x_api], 1)
# y_word_embedded = tf.concat([y_word_embedded, x_api_extend], 2)
X_seq_len = tf.placeholder(tf.int32, [None])
Y_seq_len = tf.placeholder(tf.int32, [None])

# ENCODER
encoder_out, encoder_state = tf.nn.dynamic_rnn(
    cell = tf.nn.rnn_cell.BasicLSTMCell(128),
    inputs = tf.contrib.layers.embed_sequence(x_word, 771, 20),
    sequence_length = X_seq_len,
    dtype = tf.float32)

# ATTENTION
attention_mechanism = tf.contrib.seq2seq.LuongAttention(
    num_units = 128,
    memory = encoder_out,
    memory_sequence_length = X_seq_len)

decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
    cell = tf.nn.rnn_cell.BasicLSTMCell(128),
    attention_mechanism = attention_mechanism,
    attention_layer_size = 128)

# DECODER
decoder_embedding = tf.Variable(tf.random_uniform([771, 128], -1.0, 1.0))
projection_layer = Dense(771)

training_helper = tf.contrib.seq2seq.TrainingHelper(
    inputs = y_word_embedded,  # tf.nn.embedding_lookup(decoder_embedding, y_word_in),
    sequence_length = Y_seq_len,
    time_major = False)
training_decoder = tf.contrib.seq2seq.BasicDecoder(
    cell = decoder_cell,
    helper = training_helper,
    initial_state = decoder_cell.zero_state(1234, tf.float32).clone(cell_state=encoder_state),
    output_layer = projection_layer)
training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
    decoder = training_decoder,
    impute_finished = True,
    maximum_iterations = 20)

print 'END!!!'
