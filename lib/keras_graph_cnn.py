import tensorflow as tf

from keras import backend as K
from keras.layers import Layer
import numpy as np
import scipy


class GraphConvolution(Layer):

    def __init__(self, filter_size, pooling, poly_k, L=[], bias_per_vertex=False,
                 pool_type='max', activation=None, **kwargs):

#        self.F_0 = input_channels
        self.F_1 = filter_size
#        self.M_0 = input_nodes
        self.K = poly_k
        self.p_1 = pooling
        self.L = L
        self.output_dim = ()
        self.bias_per_vertex = bias_per_vertex

        if activation is None:
            self.activation = tf.nn.relu
        else:
            self.activation = activation

        if pool_type == 'max':
            self.poolf = tf.nn.max_pool
        elif pool_type == 'average' or pool_type == 'avg':
            self.poolf = tf.nn.avg_pool
        else:
            raise ValueError('pool_type not set to "max" or "avg"')

        super(GraphConvolution, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.F_0 = input_shape[2]
        self.M_0 = input_shape[1]
        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.F_0 * self.K, self.F_1),
                                      initializer='uniform',
                                      trainable=True)

        if self.bias_per_vertex:
            self.bias = self.add_weight(name='bias', shape=(1, self.M_0, self.F_1), initializer='uniform',
                                        trainable=True)
        else:
            self.bias = self.add_weight(name='bias', shape=(1, 1, self.F_1), initializer='uniform', trainable=True)
        super(GraphConvolution, self).build(input_shape)  # Be sure to call this at the end

    def rescale_L(self, L, lmax=2):
        """Rescale the Laplacian eigenvalues in [-1,1]."""
        M, M = L.shape
        I = scipy.sparse.identity(M, format='csr', dtype=L.dtype)
        L /= lmax / 2
        L -= I
        return L

    def chebyshev5(self, x, L, Fout, K_depth):
        N, M, Fin = K.int_shape(x)
        # Rescale Laplacian and store as a TF sparse tensor. Copy to not modify the shared L.
        L = scipy.sparse.csr_matrix(L)
        L = self.rescale_L(L, lmax=2)
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data, L.shape)
        L = tf.sparse_reorder(L)
        # Transform to Chebyshev basis
        x0 = tf.transpose(x, perm=[1, 2, 0])
        x0 = tf.reshape(x0, [M, -1])
        x = tf.expand_dims(x0, 0)

        def concat(x, x_):
            x_ = tf.expand_dims(x_, 0)
            return tf.concat([x, x_], axis=0)

        if K_depth > 1:
            x1 = tf.sparse_tensor_dense_matmul(L, x0)
            x = concat(x, x1)
        for k in range(2, K_depth):
            x2 = 2 * tf.sparse_tensor_dense_matmul(L, x1) - x0
            x = concat(x, x2)
            x0, x1 = x1, x2
        x = tf.reshape(x, [K_depth, M, Fin, -1])
        x = tf.transpose(x, perm=[3, 1, 2, 0])
        x = tf.reshape(x, [-1, Fin * K_depth])
        x = tf.matmul(x, self.kernel)
        return tf.reshape(x, [-1, M, Fout])

    def pool(self, x, p):
        if p > 1:
            x = tf.expand_dims(x, 3)
            x = self.poolf(x, ksize=[1, p, 1, 1], strides=[1, p, 1, 1], padding='SAME')
            # tf.maximum
            return tf.squeeze(x, [3])
        else:
            return x

    def call(self, x):
        if len(x.get_shape()) != 3:
            x = tf.expand_dims(x, 2)
        x = self.chebyshev5(x, self.L, self.F_1, self.K)
        x = self.activation(x + self.bias)
        x = self.pool(x, self.p_1)
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.M_0 / self.p_1, self.F_1)

