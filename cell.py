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
            inputs_state = tf.concat([inputs, state], 1)
            W1 = tf.get_variable('weight1', [inputs_state.get_shape().as_list()[1], self._num_units], tf.float32)
            b1 = tf.get_variable('bias1', self._num_units, tf.float32)
            new_state = self._activation(tf.matmul(inputs_state, W1)+b1)
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
            a=1
            #We start with bias of 1.0 to not reset and not update.
            #todo: implement the new_h calculation given inputs and state
        new_h = []
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
            inputs_state = tf.concat([inputs, h], 1)
            W1 = tf.get_variable('weight1', [inputs_state.get_shape().as_list()[1], 4*self._num_units], tf.float32)
            b1 = tf.get_variable('bias1', 4*self._num_units, tf.float32)
            new_state = tf.matmul(inputs_state, W1)+b1
            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            arrays = tf.split(value=new_state, num_or_size_splits=4, axis=1)
            input_gate = arrays[0]
            new_input = arrays[1]
            forget_gate = arrays[2]
            output_gate = arrays[3]

            new_c = c * tf.sigmoid(forget_gate + self._forget_bias) + tf.sigmoid(input_gate) * self._activation(new_input)
            new_h = self._activation(new_c) * tf.sigmoid(output_gate)

            return new_h, (new_c, new_h)
