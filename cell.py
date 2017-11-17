import tensorflow as tf

class BasicRNNCell(tf.contrib.rnn.RNNCell):

    def __init__(self, num_units, activation=tf.tanh, reuse=None):
        self._num_units = num_units
        self._activation = activation
        self._reuse = reuse

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or "basic_rnn_cell", reuse=self._reuse):
            #todo: implement the new_state calculation given inputs and state
            inputs_state = tf.concat([inputs, state], 1)    #inputs和state拼在一起，效率提升10%以上
            W = tf.get_variable('weight1', [inputs_state.shape[1], self._num_units], tf.float32)
            b = tf.get_variable('bias1', self._num_units, tf.float32)
            new_state = self._activation(tf.matmul(inputs_state, W)+b)
        return new_state, new_state

class GRUCell(tf.contrib.rnn.RNNCell):
    '''Gated Recurrent Unit cell (http://arxiv.org/abs/1406.1078).'''

    def __init__(self, num_units, activation=tf.tanh, reuse=None):
        self._num_units = num_units
        self._activation = activation
        self._reuse = reuse

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or "gru_cell", reuse=self._reuse):
            inputs_state = tf.concat([inputs, state], 1)
            W1 = tf.get_variable('weight1', [inputs_state.shape[1], 2*self._num_units], tf.float32)
            b1 = tf.get_variable('bias1', 2*self._num_units, tf.float32)
            z_r = tf.matmul(inputs_state, W1)+b1
            arrays = tf.split(value=z_r, num_or_size_splits=2, axis=1) 
            z = tf.sigmoid(arrays[0])
            r = tf.sigmoid(arrays[1])

            st_1 = tf.multiply(state, r)
            inputs_st_1 = tf.concat([inputs, st_1], 1)
            W2 = tf.get_variable('weight2', [inputs_st_1.shape[1], self._num_units], tf.float32)
            b2 = tf.get_variable('bias2', self._num_units, tf.float32)
            h = self._activation(tf.matmul(inputs_st_1, W2)+b2)
            new_h = tf.multiply((1-z), h) + tf.multiply(z, state)
            #We start with bias of 1.0 to not reset and not update.
            #todo: implement the new_h calculation given inputs and state
        return new_h, new_h

class BasicLSTMCell(tf.contrib.rnn.RNNCell):
    '''Basic LSTM cell (http://arxiv.org/abs/1409.2329).'''

    def __init__(self, num_units, forget_bias=1.0, activation=tf.tanh, reuse=None):
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._activation = activation
        self._reuse = reuse

    @property
    def state_size(self):
        return (self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or "basic_lstm_cell", reuse=self._reuse):
            c, h = state
            #For forget_gate, we add forget_bias of 1.0 to not forget in order to reduce the scale of forgetting in the beginning of the training.
            #todo: implement the new_c, new_h calculation given inputs and state (c, h)
            inputs_state = tf.concat([inputs, h], 1)    #同样拼接
            W = tf.get_variable('weight1', [inputs_state.shape[1], 4*self._num_units], tf.float32)
            b = tf.get_variable('bias1', 4*self._num_units, tf.float32)
            new_state = tf.matmul(inputs_state, W)+b

            arrays = tf.split(value=new_state, num_or_size_splits=4, axis=1)    #将那个U合并在一起了
            i = tf.sigmoid(arrays[0])
            f = tf.sigmoid(arrays[1] + self._forget_bias)
            o = tf.sigmoid(arrays[2])
            g = self._activation(arrays[3])

            new_c = c * f + g * i
            new_h = o * self._activation(new_c)

            return new_h, (new_c, new_h)
