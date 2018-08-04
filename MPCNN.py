import tensorflow as tf
from tensorflow import contrib
import numpy as np

class MPCNN(object):
    """
    Multi-perspective CNN for sentence similarity

    input_A: placeholder, array of integers that represent sentence A
    input_B: placeholder, array of integers that represent sentence B
    input_y: placeholder, float that represent similarity score
    """

    def __init__(self, batch_size, sequence_length, embedding_size, filter_sizes, num_filters, word_vector, l2_reg_lambda=0.0):

        # basic properties
        self.sequence_length = sequence_length
        self.embedding_size = embedding_size
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.batch_size = batch_size


        # define placeholders
        self.input_1 = tf.placeholder(tf.int32, [batch_size, sequence_length], 'input_1')
        self.input_2 = tf.placeholder(tf.int32, [batch_size, sequence_length], 'input_2')
        self.input_y = tf.placeholder(tf.float32, [batch_size, 1], 'input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='Dropout_keep_prob')

        #
        l2_loss = 0.0

        with tf.name_scope('Sentence_model'):
            # embedding layer
            with tf.device("/cpu:0"), tf.name_scope('embedding'):
                W = tf.Variable(dtype=tf.float32, initial_value=word_vector, name="embedding_weight")

                # embedding layer for both two types of sentences
                self.embedded_chars_1 = tf.nn.embedding_lookup(W, self.input_1)
                self.embedded_chars_2 = tf.nn.embedding_lookup(W, self.input_2)

                # expand dimension
                self.embedded_expanded_1 = tf.expand_dims(self.embedded_chars_1, axis=-1)
                self.embedded_expanded_2 = tf.expand_dims(self.embedded_chars_2, axis=-1)

            with tf.name_scope("Block_A"):
                # block A for sentence 1
                self.pooled_output_A_1 = self.block_A(self.embedded_expanded_1, sentence=1)
                self.pooled_output_A_2 = self.block_A(self.embedded_expanded_2, sentence=2)

            with tf.name_scope("Block_B"):
                # block B for sentence 1
                self.pooled_output_B_1 = self.block_B(self.embedded_expanded_1, sentence=1)
                self.pooled_output_B_2 = self.block_B(self.embedded_expanded_2, sentence=2)

        with tf.name_scope("measurement_layer"):
            self.feah, self.feaa, self.feab = [], [], []
            for pool in ["max_pool", "min_pool", "avg_pool"]:
                with tf.name_scope('Horizon_A'):
                    self.feah.append(self.horizon_A(self.pooled_output_A_1[pool], self.pooled_output_A_2[pool]))

                with tf.name_scope('Vertical_A'):
                    self.feaa.append(self.vertical_A(self.pooled_output_A_1[pool], self.pooled_output_A_2[pool]))

                with tf.name_scope('Vertical_B'):
                    self.feab.append(self.vertical_B(self.pooled_output_B_1[pool], self.pooled_output_B_2[pool]))
            self.feah = tf.concat(self.feah, axis=1)
            self.feaa = tf.concat(self.feaa, axis=1)
            self.feab = tf.concat(self.feab, axis=1)

        self.h_pool_flat = tf.concat([self.feah, self.feaa, self.feab], axis=1)
        total_num_filters = self.h_pool_flat.shape[1]  # 3 * (self.num_filters * self.filter_sizes * 2 + self.num_filters)

        # dropout
        with tf.name_scope('Dropout'):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)


        # fully connected layer
        with tf.name_scope('Fully-connected'):
            out1 = contrib.layers.fully_connected(self.h_drop, num_outputs=150, activation_fn=tf.nn.tanh)
            out2 = contrib.layers.fully_connected(self.h_drop, num_outputs=150)

        # output
        with tf.name_scope("output"):
            W = tf.Variable(tf.truncated_normal([150, 1], stddev=0.1), name='W')
            b = tf.Variable(tf.constant(0.1, shape=[1], name="b"))

            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)

            self.scores = tf.nn.xw_plus_b(out2, W, b, name='scores')

        # scores and predictions
        with tf.name_scope('loss'):
            self.loss = tf.square(self.scores - self.input_y)
            self.loss = tf.reduce_mean(self.loss) + l2_reg_lambda * l2_loss

        # pearson prediction
        with tf.name_scope('Pearson'):
            numerator = tf.reduce_mean(self.scores * self.input_y) - \
                        tf.reduce_mean(self.scores) * tf.reduce_mean(self.input_y)

            denominator = tf.sqrt(tf.reduce_mean(tf.square(self.scores)) - tf.square(tf.reduce_mean(self.scores))) * \
                          tf.sqrt(tf.reduce_mean(tf.square(self.input_y)) - tf.square(tf.reduce_mean(self.input_y)))

            self.pearson = numerator / denominator

    # Blocks
    def block_A(self, input, sentence=1):
        """
        block A
        :param input: Variable, embedded_chars_expanded, [None, sequence_length, embedded_size, 1]
        :param sentence: indicate which sentence is processed here
        :return: dictionary for pooling output [batch_size, filter_sizes, 1, num_filters]
        """
        with tf.name_scope("Block_A_sent_%s" % sentence):
            pooled_output = {}
            for i, filter_size in enumerate(range(1, self.filter_sizes+1)):
                with tf.name_scope("conv_filter-%s" % (i + 1)):
                    filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]

                    # W: the initial value of filter map
                    W = tf.Variable(dtype=tf.float32, initial_value=tf.truncated_normal(shape=filter_shape, stddev=0.1), name="W")

                    # b: the initial value of bias
                    b = tf.Variable(dtype=tf.float32, initial_value=tf.constant(0.1, shape=[self.num_filters]), name='b')

                    # convolutional layer
                    conv = tf.nn.conv2d(input,
                                        filter=W,
                                        strides=[1, 1, 1, 1],
                                        padding="VALID")

                    # apply non-linearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')

                    kernel_size = [1, self.sequence_length - filter_size + 1, 1, 1]
                    max_pooled = self.max_pool(h, ksize=kernel_size)
                    min_pooled = self.min_pool(h, ksize=kernel_size)
                    avg_pooled = self.avg_pool(h, ksize=kernel_size)

                    pooled_output['max_pool'] = pooled_output.get('max_pool', []) + [max_pooled]
                    pooled_output['min_pool'] = pooled_output.get('min_pool', []) + [min_pooled]
                    pooled_output['avg_pool'] = pooled_output.get('avg_pool', []) + [avg_pooled]

            # concatenate pooling outputs for any filter sizes
            pooled_output['max_pool'] = tf.concat(pooled_output['max_pool'], axis=1)
            pooled_output['min_pool'] = tf.concat(pooled_output['min_pool'], axis=1)
            pooled_output['avg_pool'] = tf.concat(pooled_output['avg_pool'], axis=1)
            return pooled_output

    def block_B(self, input, sentence=1):
        """
        block B
        :param input: Variable, embedded_chars_expanded, [None, sequence_length, embedded_size, 1]
        :param sentence: indicate which sentence is processed here
        :return: dictionary for pooling output [batch_size, filter_sizes, embedded_size, num_filters]
        """
        with tf.name_scope("Block_B_sent-%s" % sentence):
            pooled_output = {}
            for i, filter_size in enumerate(range(1, self.filter_sizes+1)):
                with tf.name_scope("conv_filter_%s" % (i+1)):
                    filter_shape = [filter_size, 1, 1, self.num_filters]

                    # W: the initial value of filter map
                    W = tf.Variable(tf.truncated_normal(shape=filter_shape, stddev=0.1), name="W")

                    # b: the initial value of bias
                    b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name='b')

                    max_pool, min_pool, avg_pool = [], [], []
                    for i in range(input.shape[-2]):
                        tmp = tf.expand_dims(input[:, :, i, :], axis=-2)
                        conv = tf.nn.conv2d(tmp, W,
                                            strides=[1,1,1,1],
                                            padding="VALID")

                        # apply non-linearity
                        h = tf.nn.relu(tf.nn.bias_add(conv, b))

                        # pooling output
                        kernel_size = [1, self.sequence_length-filter_size+1, 1, 1]
                        max_pooled = self.max_pool(h, kernel_size)
                        min_pooled = self.min_pool(h, kernel_size)
                        avg_pooled = self.avg_pool(h, kernel_size)

                        max_pool.append(max_pooled)
                        min_pool.append(min_pooled)
                        avg_pool.append(avg_pooled)

                    # concatenate 'dim' elements of pooling output where dim = self.embedded_size
                    max_pool = tf.concat(max_pool, axis=2)
                    min_pool = tf.concat(min_pool, axis=2)
                    avg_pool = tf.concat(avg_pool, axis=2)

                    # pooling output for all filter_sizes
                    pooled_output['max_pool'] = pooled_output.get('max_pool', []) + [max_pool]
                    pooled_output['min_pool'] = pooled_output.get('min_pool', []) + [min_pool]
                    pooled_output['avg_pool'] = pooled_output.get('avg_pool', []) + [avg_pool]

            # concatenate pooling outputs for any filter sizes
            pooled_output['max_pool'] = tf.concat(pooled_output['max_pool'], axis=1)
            pooled_output['min_pool'] = tf.concat(pooled_output['min_pool'], axis=1)
            pooled_output['avg_pool'] = tf.concat(pooled_output['avg_pool'], axis=1)
            return pooled_output

    # pooling functions
    def max_pool(self, input, ksize):
        """
        The maximum value pooling layer
        """
        return tf.nn.max_pool(input, ksize=ksize, strides=[1, 1, 1, 1], padding="VALID")

    def min_pool(self, input, ksize):
        """
        The minimum value pooling layer
        """
        return -1 * tf.nn.max_pool(-1*input, ksize=ksize, strides=[1, 1, 1, 1], padding="VALID")

    def avg_pool(self, input, ksize):
        """
        The mean value pooling layer
        """
        return tf.nn.avg_pool(input, ksize=ksize, strides=[1, 1, 1, 1], padding="VALID")

    # distance method
    def cosine_sim(self, array1, array2):
        """
        compute cosine similarity between two vectors
        :param array1: array 1
        :param array2: array 2
        :return: the similarity score between 0 and 1
        """
        array1_norm = tf.nn.l2_normalize(array1, axis=0)
        array2_norm = tf.nn.l2_normalize(array2, axis=0)
        return 1-tf.losses.cosine_distance(array1_norm, array2_norm, axis=0)


    # compute distances
    def horizon_A(self, input1, input2):    # feah
        """
        input: [batch_size, filter_sizes, 1, num_filters]
        :return: [batch_size, num_filters]
        """
        input1 = tf.reshape(input1, shape=[-1, self.filter_sizes, self.num_filters])
        input2 = tf.reshape(input2, shape=[-1, self.filter_sizes, self.num_filters])

        def func(m1, m2):
            return [[self.cosine_sim(m1[:, i], m2[:, i]) for i in range(self.num_filters)]]

        output = tf.concat([func(input1[i], input2[i]) for i in range(self.batch_size)], axis=0)
        return output

    def vertical_A(self, input1, input2):
        """
        input: [batch_size, filter_sizes, 1, num_filters]
        :return: [batch_size, num_filters * filter_sizes]
        """
        input1 = tf.reshape(input1, shape=[-1, self.filter_sizes, self.num_filters])
        input2 = tf.reshape(input2, shape=[-1, self.filter_sizes, self.num_filters])

        def func(m1, m2):
            lis = []
            for i in range(self.num_filters):
                for j in range(self.filter_sizes):
                    lis.append(self.cosine_sim(m1[:, i], m2[:,j]))
            return [lis]

        output = tf.concat([func(input1[i], input2[i]) for i in range(self.batch_size)], axis=0)
        return output

    def vertical_B(self, input1, input2):
        """
        input: [batch_size, filter_sizes, embedded_size, num_filters]
        :return: [batch_size, filter_sizes * num_filters]
        """

        def func(m1, m2):
            lis = []
            for i in range(self.filter_sizes):
                lis += [self.cosine_sim(m1[i, :, j], m2[i, :, j]) for j in range(self.num_filters)]
            return [lis]
        output = tf.concat([func(input1[i], input2[i]) for i in range(self.batch_size)], axis=0)
        return output
"""
if __name__ == '__main__':
    batch_size = 2
    word_vector = np.ones((6, 6))
    sequence_length = 6
    embedding_size = 6
    filter_sizes = 4
    num_filters = 4

    test = MPCNN(batch_size, sequence_length, embedding_size, filter_sizes, num_filters, word_vector)

    init = tf.global_variables_initializer()
    sess = tf.Session()

    sess.run(init)

    feet_dict = {test.input_1: np.array([[0, 1, 2, 3, 4, 5],[0,1,2,3,4,5]]).reshape([2, 6]),
                 test.input_2: np.array([[0, 1, 2, 3, 4, 5],[0,1,2,3,4,5]]).reshape([2, 6])}

    sess.run(test.pooled_output_A_1, feed_dict=feet_dict)
"""