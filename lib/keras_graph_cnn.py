import tensorflow as tf

from keras import backend as K
from keras.layers import Layer
import numpy as np
import scipy
import graph

class MyLayer(Layer):

    def __init__(self, input_nodes, input_channels, filter_size, pooling, poly_k, L = [], brelu='b1relu', **kwargs):

        self.F_0 = input_channels
        self.F_1 = filter_size
        self.M_0 = input_nodes
        self.K = poly_k
        self.p_1 = pooling
        self.L = L

        self.output_dim = ()
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.F_0 *self.K, self.F_1),
                                      initializer='uniform',
                                      trainable=True)
        self.bias = self.add_weight(name='bias', shape=(1,1,self.F_1),  initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this at the end

    def chebyshev5(self, x, L, Fout, K):
        N, M, Fin = x.get_shape()
        N, M, Fin = int(N), int(M), int(Fin)
        # Rescale Laplacian and store as a TF sparse tensor. Copy to not modify the shared L.
        L = scipy.sparse.csr_matrix(L)
        L = graph.rescale_L(L, lmax=2)
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data, L.shape)
        L = tf.sparse_reorder(L)
        # Transform to Chebyshev basis
        x0 = tf.transpose(x, perm=[1, 2, 0])  # M x Fin x N
        x0 = tf.reshape(x0, [M, Fin*N])  # M x Fin*N
        x = tf.expand_dims(x0, 0)  # 1 x M x Fin*N
        def concat(x, x_):
            x_ = tf.expand_dims(x_, 0)  # 1 x M x Fin*N
            return tf.concat([x, x_], axis=0)  # K x M x Fin*N
        if K > 1:
            x1 = tf.sparse_tensor_dense_matmul(L, x0)
            x = concat(x, x1)
        for k in range(2, K):
            x2 = 2 * tf.sparse_tensor_dense_matmul(L, x1) - x0  # M x Fin*N
            x = concat(x, x2)
            x0, x1 = x1, x2
        x = tf.reshape(x, [K, M, Fin, N])  # K x M x Fin x N
        x = tf.transpose(x, perm=[3,1,2,0])  # N x M x Fin x K
        x = tf.reshape(x, [N*M, Fin*K])  # N*M x Fin*K
        # Filter: Fin*Fout filters of order K, i.e. one filterbank per feature pair.
      #  W = self._weight_variable([Fin*K, Fout], regularization=False)
        x = tf.matmul(x, self.kernel)  # N*M x Fout
        return tf.reshape(x, [N, M, Fout])  # N x M x Fout

    def b1relu(self, x):
        """Bias and ReLU. One bias per filter."""
        N, M, F = x.get_shape()
        return tf.nn.relu(x + self.bias)

    def mpool1(self, x, p):
        """Max pooling of size p. Should be a power of 2."""
        if p > 1:
            x = tf.expand_dims(x, 3)  # N x M x F x 1
            x = tf.nn.max_pool(x, ksize=[1,p,1,1], strides=[1,p,1,1], padding='SAME')
            #tf.maximum
            return tf.squeeze(x, [3])  # N x M/p x F
        else:
            return x

    def call(self, x):
        if len(x.get_shape()) != 3:
            x = tf.expand_dims(x, 2)
            assert(x.get_shape()[1] == self.M_0)
            assert(x.get_shape()[2] == self.F_1)

        x = self.chebyshev5(x, self.L, self.F_1, self.K)
        x = self.brelu(x)
        x = self.mpool1(x, self.p_1)
        return x


    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.M_0 / self.p_1 ,self.F_1)
