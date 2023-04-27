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
            l2_reg_lambda = 0, 
            filter_sizes = [3, 4, 5, 6], 
            mode = 0
    ):
        if mode == 4 or mode == 5:
            self.input_x_char = tf.keras.Input(name="input_x_char", shape=(None, None, None), dtype=tf.int32)
            self.input_x_char_pad_idx = tf.keras.Input(name="input_x_char_pad_idx", shape=(None, None, None, embedding_size), dtype=tf.float32)
        if mode == 2 or mode == 3 or mode == 4 or mode == 5:
            self.input_x_word = tf.keras.Input(name="input_x_word", shape=(None, None), dtype=tf.int32)
        if mode == 1 or mode == 3 or mode == 5:
            self.input_x_char_seq = tf.keras.Input(name="input_x_char_seq", shape=(None, None), dtype=tf.int32)
        self.input_y = tf.keras.Input(name="input_y", shape=(None, 2), dtype=tf.float32)
        self.dropout_keep_prob = tf.compat.v1.placeholder(tf.float32, name="dropout_keep_prob")

        l2_loss = tf.constant(0.0)
        with tf.name_scope("embedding"):
            initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)
            if mode == 4 or mode == 5:
                self.char_w = tf.Variable(initializer([char_ngram_vocab_size, embedding_size]), name = "char_emb_w")
            if mode == 2 or mode == 3 or mode == 4 or mode == 5:
                self.word_w = tf.Variable(initial_value = initializer([word_ngram_vocab_size, embedding_size]), name = "word_emb_w")
            if mode == 1 or mode == 3 or mode == 5:
                self.char_seq_w = tf.Variable(initializer([char_vocab_size, embedding_size]), name = "char_seq_emb_w")
            
            if mode == 4 or mode == 5:
                self.embedded_x_char = tf.nn.embedding_lookup(self.char_w, self.input_x_char)
                self.embedded_x_char = tf.multiply(self.embedded_x_char, self.input_x_char_pad_idx)
            if mode == 2 or mode == 3 or mode == 4 or mode == 5:
                self.embedded_x_word = tf.nn.embedding_lookup(self.word_w, self.input_x_word)
            if mode == 1 or mode == 3 or mode == 5:
                self.embedded_x_char_seq = tf.nn.embedding_lookup(self.char_seq_w, self.input_x_char_seq)

            if mode == 4 or mode == 5:
                self.sum_ngram_x_char = tf.reduce_sum(self.embedded_x_char, 2)
                self.sum_ngram_x = tf.add(self.sum_ngram_x_char, self.embedded_x_word)
            
            if mode == 4 or mode == 5:
                self.sum_ngram_x_expanded = tf.expand_dims(self.sum_ngram_x, -1)
            if mode == 2 or mode == 3:
                self.sum_ngram_x_expanded = tf.expand_dims(self.embedded_x_word, -1)
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
                    pooled = tf.compat.v1.nn.max_pool(
                        h,
                        ksize = [1, word_seq_len-filter_size+1, 1, 1],
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
        if mode == 1 or mode == 3 or mode == 5:
            pooled_char_x = []
            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope("char_conv_maxpool_%s" % filter_size):
                    filter_shape = [filter_size, embedding_size, 1, 256]
                    b = tf.Variable(tf.constant(0.1, shape=[256]), name="b")
                    w = tf.Variable(tf.random.truncated_normal(filter_shape, stddev=0.1), name="w")
                    conv = tf.nn.conv2d(
                        self.char_x_expanded,
                        w,
                        strides = [1, 1, 1, 1],
                        padding = "VALID",
                        name = "conv"
                    )
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    pooled = tf.compat.v1.nn.max_pool(
                        h,
                        ksize = [1, char_seq_len-filter_size+1, 1, 1],
                        strides = [1, 1, 1, 1],
                        padding = "VALID",
                        name = "pool"
                    )
                    pooled_char_x.append(pooled)
            num_filters_total = 256 * len(filter_sizes)
            self.h_char_pool = tf.concat(pooled_char_x, 3)
            self.char_x_flat = tf.reshape(self.h_char_pool, [-1, num_filters_total], name="pooled_char_x")
            self.char_h_drop = tf.nn.dropout(self.char_x_flat, self.dropout_keep_prob, name="dropout_char_x")

        # Concat word and char branch
        if mode == 3 or mode == 5:
            with tf.name_scope("word_char_concat"):
                initializer = tf.keras.initializers.GlorotNormal()
                ww = tf.Variable(initializer(shape=(num_filters_total, 512)), name="ww")
                bw = tf.Variable(tf.constant(0.1, shape=[512]), name="bw")
                l2_loss += tf.nn.l2_loss(ww)
                l2_loss += tf.nn.l2_loss(bw)
                word_output = tf.compat.v1.nn.xw_plus_b(self.h_drop, ww, bw)

                wc = tf.Variable(initializer(shape=(num_filters_total, 512)), name="wc")
                bc = tf.Variable(tf.constant(0.1, shape=[512]), name="bc")
                l2_loss += tf.nn.l2_loss(wc)
                l2_loss += tf.nn.l2_loss(bc)
                char_output = tf.compat.v1.nn.xw_plus_b(self.char_h_drop, wc, bc)

                self.conv_output = tf.concat([word_output, char_output], 1)
        elif mode == 2 or mode == 4:
            self.conv_output = self.h_drop
        elif mode == 1:
            self.conv_output = self.char_h_drop

        # Relu and FC
        with tf.name_scope("output"):
            initializer = tf.keras.initializers.GlorotNormal()
            w0 = tf.Variable(initializer(shape=(1024, 512)), name="w0")
            b0 = tf.Variable(tf.constant(0.1, shape=[512]), name="b0") 
            l2_loss += tf.nn.l2_loss(w0) 
            l2_loss += tf.nn.l2_loss(b0) 
            output0 = tf.nn.relu(tf.matmul(self.conv_output, w0) + b0)

            w1 = tf.Variable(initializer(shape=(512, 256)), name="w1")
            b1 = tf.Variable(tf.constant(0.1, shape=[256]), name='b1')
            l2_loss += tf.nn.l2_loss(w1)
            l2_loss += tf.nn.l2_loss(b1)
            output1 = tf.nn.relu(tf.matmul(output0, w1) + b1)

            w2 = tf.Variable(initializer(shape=(256, 128)), name="w2")
            b2 = tf.Variable(tf.constant(0.1, shape=[128]), name="b2")
            l2_loss += tf.nn.l2_loss(w2)
            l2_loss += tf.nn.l2_loss(b2)
            output2 = tf.nn.relu(tf.matmul(output1, w2) + b2)

            w = tf.Variable(initializer(shape=(128, 2)), name="w")
            b = tf.Variable(tf.constant(0.1, shape=[2]), name="b")
            l2_loss += tf.nn.l2_loss(w)
            l2_loss += tf.nn.l2_loss(b)

            self.scores = tf.compat.v1.nn.xw_plus_b(output2, w, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        with tf.name_scope("accuracy"):
            correct_preds = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_preds, "float"), name="accuracy")
