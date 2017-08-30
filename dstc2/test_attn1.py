import tensorflow as tf
from tensorflow.python.layers.core import Dense

# INPUT
X = tf.placeholder(tf.int32, [None, None])
Y = tf.placeholder(tf.int32, [None, None])
X_seq_len = tf.placeholder(tf.int32, [None])
Y_seq_len = tf.placeholder(tf.int32, [None])













# ENCODER
encoder_out, encoder_state = tf.nn.dynamic_rnn(
    cell = tf.nn.rnn_cell.BasicLSTMCell(128),
    inputs = tf.contrib.layers.embed_sequence(X, 10000, 128),
    sequence_length = X_seq_len,
    dtype = tf.float32)
print X_seq_len
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
Y_vocab_size = 10000
decoder_embedding = tf.Variable(tf.random_uniform([Y_vocab_size, 128], -1.0, 1.0))
projection_layer = Dense(Y_vocab_size)

training_helper = tf.contrib.seq2seq.TrainingHelper(
    inputs = tf.nn.embedding_lookup(decoder_embedding, Y),
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
    maximum_iterations = tf.reduce_max(Y_seq_len))
training_logits = training_decoder_output.rnn_output

# LOSS
masks = tf.sequence_mask(Y_seq_len, tf.reduce_max(Y_seq_len), dtype=tf.float32)
loss = tf.contrib.seq2seq.sequence_loss(logits = training_logits, targets = Y, weights = masks)

# BACKWARD
params = tf.trainable_variables()
gradients = tf.gradients(loss, params)
clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
train_op = tf.train.AdamOptimizer().apply_gradients(zip(clipped_gradients, params))