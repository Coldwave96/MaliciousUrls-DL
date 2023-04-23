import tensorflow as tf


class UrlNet(object):
    def __init__(self, char_ngram_vocab_size, word_ngram_vocab_size, char_vocab_size, word_seq_len, char_seq_len, embedding_size, 
                 l2_reg_lambda=0, filter_sizes=[3, 4, 5, 6], mode=0):
        if mode == 4 or mode == 5:
            self.input_x_char = tf.keras.Input(name="input_x_char", type_spec=tf.RaggedTensorSpec(shape=[None, None, None], dtype=tf.int32, ragged_rank=1))
            self.input_x_char_pad_idx = tf.keras.Input(name="input_x_char_pad_idx", type_spec=tf.RaggedTensorSpec(shape=[None, None, None, embedding_size], dtype=tf.float32, ragged_rank=1))
        if mode == 2 or mode == 3 or mode == 4 or mode == 5:
            self.input_x_word = tf.keras.Input(name="input_x_word", type_spec=tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int32, ragged_rank=1))
        if mode == 1 or mode == 3 or mode == 5:
            self.input_x_char_seq = tf.keras,input(name="inpur_x_char_seq", type_sepc=tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int32, ragged_rank=1))
        
        self.input_y = tf.keras.Input(name="input_y", type_spec=tf.RaggedTensorSpec(shape=[None, 2], dtype=tf.float32, ragged_rank=1))
        self.dropout_keep_prob = tf.keras.Input(name="dropout_keep_prob", type_spec=tf.RaggedTensorSpec(dtype=tf.float32))
