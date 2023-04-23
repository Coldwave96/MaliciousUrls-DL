import tensorflow as tf


class UrlNet(object):
    def __init__(
            self, 
            char_ngram_vocab_size, 
            word_ngram_vocab_size, 
            char_vocab_size, 
            word_seq_len, 
            char_seq_len, 
            embedding_size, 
            l2_reg_lambda=0, 
            filter_sizes=[3, 4, 5, 6], 
            mode=0
    ):
        if mode == 4 or mode == 5:
            self.input_x_char = tf.keras.Input(
                name = "input_x_char", 
                type_spec = tf.RaggedTensorSpec(shape=[None, None, None], dtype=tf.int32, ragged_rank=1)
            )
            self.input_x_char_pad_idx = tf.keras.Input(
                name="input_x_char_pad_idx", 
                type_spec=tf.RaggedTensorSpec(shape=[None, None, None, embedding_size], dtype=tf.float32, ragged_rank=1)
            )
        if mode == 2 or mode == 3 or mode == 4 or mode == 5:
            self.input_x_word = tf.keras.Input(
                name="input_x_word", 
                type_spec=tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int32, ragged_rank=1)
            )
        if mode == 1 or mode == 3 or mode == 5:
            self.input_x_char_seq = tf.keras,input(
                name="inpur_x_char_seq", 
                type_sepc=tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int32, ragged_rank=1)
            )
        
        self.input_y = tf.keras.Input(
            name="input_y", 
            type_spec=tf.RaggedTensorSpec(shape=[None, 2], dtype=tf.float32, ragged_rank=1)
        )
        self.dropout_keep_prob = tf.keras.Input(
            name="dropout_keep_prob", 
            type_spec=tf.RaggedTensorSpec(dtype=tf.float32)
        )

        l2_loss = tf.constant(0.0)
        with tf.name_scope("embedding"):
            if mode == 4 or mode == 5:
                self.char_w = tf.Variable(
                    tf.random_uniform_initializer([char_ngram_vocab_size, embedding_size], -1.0, 1.0), 
                    name="char_emb_w"
                )
            if mode == 2 or mode == 3 or mode == 4 or mode == 5:
                self.word_w = tf.Variable(
                    tf.random_uniform_initializer([word_ngram_vocab_size, embedding_size], -1.0, 1.0), 
                    name="word_emb_w"
                )
            if mode == 1 or mode == 3 or mode == 5:
                self.char_seq_w = tf.Variable(
                    tf.random_uniform_initializer([char_vocab_size, embedding_size], -1.0, 1.0), 
                    name="char_seq_emb_w"
                )
            
            if mode == 4 or mode == 5:
                self.embedded_x_char = tf.nn.embedding_lookup(self.char_w, self.input_x_char)
                self.embedded_x_char = tf.multiply(self.embedded_x_char, self.input_x_char_pad_idx)
            if mode == 2 or mode == 3 or mode == 4 or mode == 5:
                self.embedded_x_word = tf.nn.embedding_lookup(self.word_w, self.input_x_word)
            if mode == 1 or mode == 3 or mode == 5:
                self.embedded_x_char_seq = tf.nn.embedding_lookup(self.char_seq_w, self.input_x_char_seq)

            if mode == 4 or mode == 5:
                self.sum_ngram_x_char = tf.reduce_sum(self.sum_ngram_x, 2)
                self.sum_ngram_x = tf.add(self.sum_ngram_x_char, self.embedded_x_word)
            
            if mode == 4 or mode == 5:
                self.sum_ngram_x_expanded = tf.expand_dims(self.sum_ngram_x, -1)
            if mode == 2 or mode == 3:
                self.sum_ngram_x_expanded == tf.expand_dims(self.embedded_x_word, -1)
            if mode == 1 or mode == 3 or mode == 5:
                self.char_x_expanded = tf.expand_dims(self.embedded_x_char_seq, -1)
        
        # Word convolution layer
        if mode == 2 or mode == 3 or mode == 4 or mode == 5:
            pooled_x = []

            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope("conv_maxpool_%s" % filter_size):
                    filter_shape = [filter_size, embedding_size, 1, 256]
                    b = tf.Variable(tf.constant(0.1, shape=[256]), name="b")
                    w = tf.Variable(tf.random.truncated_normal(filter_shape, stddev=0.1), name="w")
                    conv = tf.nn.conv2d(
                        self.sum_ngram_x_expanded,
                        w,
                        strides = [1, 1, 1, 1],
                        padding = "VALID",
                        name = "conv"
                    )
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    pooled = tf.nn.max_pool(
                        h,
                        ksize = [1, word_seq_len - filter_size + 1, 1, 1],
                        strides = [1, 1, 1, 1],
                        padding = "VALID",
                        name = "pool"
                    )
                    pooled_x.append(pooled)

            num_filters_total = 256 * len(filter_sizes)
            self.h_pool = tf.concat(pooled_x, 3)
            self.x_flat = tf.reshape(self.h_pool, [-1, num_filters_total], name="pooled_x")
            self.h_drop = tf.nn.dropout(self.x_flat, self.dropout_keep_prob, name="dropout_x")
        
        # Char convolution layer

