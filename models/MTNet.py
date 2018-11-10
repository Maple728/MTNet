import tensorflow as tf

class MTNet:
    def __init__(self, config, scope = 'MTNet'):
        self.config = config
        with tf.variable_scope(scope, reuse = False):
            X = tf.placeholder(tf.float32, shape = [None, self.config.n, self.config.T, self.config.D])
            Q = tf.placeholder(tf.float32, shape = [None, self.config.T, self.config.D])
            Y = tf.placeholder(tf.float32, shape = [None, self.config.D])

            input_keep_prob = tf.placeholder(tf.float32)
            output_keep_prob = tf.placeholder(tf.float32)

            # ------- no-linear component----------------
            # <batch_size, n, en_rnn_hidden_sizes>
            m_is = self.__encoder(X, self.config.n, scope = 'm')
            c_is = self.__encoder(X, self.config.n, scope = 'c')
            # <batch_size, 1, en_rnn_hidden_sizes>
            u = self.__encoder(tf.reshape(Q, shape = [-1, 1, self.config.T, self.config.D]), 1, scope = 'in')

            p_is = tf.matmul(m_is, tf.transpose(u, perm = [0, 2, 1]))
            p_is = tf.squeeze(p_is, axis = [-1])
            p_is = tf.nn.softmax(p_is)
            # <batch_size, n, 1>
            p_is = tf.expand_dims(p_is, -1)

            # <batch_size, n, en_rnn_hidden_sizes> = <batch_size, n, en_rnn_hidden_sizes> * <batch_size, n, 1>
            o_is = tf.multiply(c_is, p_is)

            pred_w = tf.get_variable('pred_w', shape = [self.config.en_rnn_hidden_sizes[-1] * (self.config.n + 1), self.config.D],
                                     dtype = tf.float32, initializer = tf.truncated_normal_initializer(stddev = 0.1))
            pred_b = tf.get_variable('pred_b', shape = [self.config.D],
                                     dtype = tf.float32, initializer = tf.constant_initializer(0.1))
            
            pred_x = tf.concat([o_is, u], axis = 1)
            pred_x = tf.reshape(pred_x, shape = [-1, self.config.en_rnn_hidden_sizes[-1] * (self.config.n + 1)])

            # <batch_size, D>
            y_pred = tf.matmul(pred_x, pred_w) + pred_b

            # ------------ ar component ------------
            if self.config.highway_window > 0:
                highway_ws = tf.get_variable('highway_ws', shape = [self.config.highway_window, self.config.D, self.config.D],
                                            dtype = tf.float32,
                                            initializer = tf.truncated_normal_initializer(stddev = 0.1))
                highway_b = tf.get_variable('highway_b', shape = [self.config.D], dtype = tf.float32,
                                            initializer = tf.constant_initializer(0.1))

                y_pred_l = tf.matmul(Q[:, 0], highway_ws[0]) + highway_b
                _, y_pred_l = tf.while_loop(lambda i, _ : tf.less(i, self.config.highway_window),
                                            lambda i, acc : (i + 1, tf.matmul(Q[:, i], highway_ws[i]) + y_pred_l),
                                            loop_vars = [1, y_pred_l])

                y_pred += y_pred_l

        # assignment
        self.X = X
        self.Q = Q
        self.Y = Y
        self.input_keep_prob = input_keep_prob
        self.output_keep_prob = output_keep_prob

        self.y_pred = y_pred
        self.loss = tf.losses.mean_squared_error(self.Y, y_pred)
        self.train_op = tf.train.AdamOptimizer(config.lr).minimize(self.loss)

    def __encoder(self, input_x, n, strides = [1, 1, 1, 1], padding = 'VALID', activation_func = tf.nn.relu,scope = 'default'):
        '''
            Treat batch_size dimension and n dimension as one batch_size dimension (batch_size * n).
        :param input_x:  <batch_size, n, T, D>
        :param strides:
        :param padding:
        :param scope:
        :return: the embedded of the input_x <batch_size, n, en_rnn_hidden_sizes>
        '''
        # constant
        scope = 'Encoder_' + scope
        batch_size_new = self.config.batch_size * n
        Tc = self.config.T - self.config.W + 1

        # reshape input_x : <batch_size * n, T, D, 1>
        input_x = tf.reshape(input_x, shape = [-1, self.config.T, self.config.D, 1])

        with tf.variable_scope(scope, reuse = tf.AUTO_REUSE):
            # cnn parameters
            w_conv1 = tf.get_variable('w_conv1', shape = [self.config.W, self.config.D, 1, self.config.en_conv_hidden_size], dtype = tf.float32, initializer = tf.truncated_normal_initializer(stddev = 0.1))
            b_conv1 = tf.get_variable('b_conv1', shape = [self.config.en_conv_hidden_size], dtype = tf.float32, initializer = tf.constant_initializer(0.1))

            # <batch_size_new, Tc, 1, en_conv_hidden_size>
            h_conv1 = tf.nn.conv2d(input_x, w_conv1, strides, padding = padding) + b_conv1

            # tmporal attention layer and gru layer

            # rnn inputs
            # <Tc, batch_size_new, 1, en_conv_hidden_size>
            rnn_input = tf.expand_dims( tf.transpose(h_conv1[:, :, 0, :], perm = [1, 0, 2]), -2)
            # <en_conv_hidden_size, batch_size_new, Tc>
            attr_input = tf.transpose(h_conv1[:, :, 0, :], perm = [2, 0, 1])

            # rnns
            rnns = [tf.nn.rnn_cell.GRUCell(h_size, activation = tf.nn.relu) for h_size in self.config.en_rnn_hidden_sizes]
            # dropout
            rnns = [tf.nn.rnn_cell.DropoutWrapper(rnn,
                                                  input_keep_prob = self.config.input_keep_prob,
                                                  output_keep_prob = self.config.output_keep_prob)
                    for rnn in rnns]

            if len(rnns) > 1:
                rnns = tf.nn.rnn_cell.MultiRNNCell(rnns)
            else:
                rnns = rnns[0]

            # attention layer
            # <batch_size_new, en_rnn_hidden_sizes>
            h_state = rnns.zero_state(batch_size_new, tf.float32)
            
            # attention weights
            attr_v = tf.get_variable('attr_v', shape = [Tc, 1], dtype = tf.float32, initializer = tf.truncated_normal_initializer(stddev = 0.1))
            attr_w = tf.get_variable('attr_w', shape = [self.config.en_conv_hidden_size, Tc], dtype = tf.float32, initializer = tf.truncated_normal_initializer(stddev = 0.1))
            attr_u = tf.get_variable('attr_u', shape = [Tc, Tc], dtype = tf.float32, initializer = tf.truncated_normal_initializer(stddev = 0.1))

            for t in range(Tc):
                # h(t-1) dot attr_w
                h_part = tf.matmul(h_state, attr_w)

                # en_conv_hidden_size * <batch_size_new, 1>
                e_ks = tf.TensorArray(tf.float32, self.config.en_conv_hidden_size)
                _, output = tf.while_loop(lambda i, _ : tf.less(i, self.config.en_conv_hidden_size),
                                          lambda i, output_ta : (i + 1, output_ta.write(i, tf.matmul(tf.tanh( h_part + tf.matmul(attr_input[i], attr_u) ), attr_v))),
                                          [0, e_ks])
                # <batch_size_new, en_conv_hidden_size, 1>
                e_ks = tf.transpose(output.stack(), perm = [1, 0, 2])

                e_ks = tf.reshape(e_ks, shape = [-1, self.config.en_conv_hidden_size])
                # <batch_size, en_conv_hidden_size>
                a_ks = tf.nn.softmax(e_ks)

                x_t = tf.matmul( rnn_input[t], tf.matrix_diag(a_ks))
                # <batch_size, en_conv_hidden_size>
                x_t = tf.reshape(x_t, shape = [-1, self.config.en_conv_hidden_size])

                h_state = rnns(x_t, h_state)
                h_state = h_state[0]
                
            return tf.reshape(h_state, shape = [-1, n, self.config.en_rnn_hidden_sizes[-1]])

    def train(self, one_batch, sess):
        _, loss, pred = sess.run([self.train_op, self.loss, self.y_pred], feed_dict = {self.X : one_batch[0],
                                                                                       self.Q : one_batch[1],
                                                                                       self.Y : one_batch[2],
                                                                                       self.input_keep_prob : self.config.input_keep_prob,
                                                                                       self.output_keep_prob : self.config.output_keep_prob})
        return loss, pred

    def predict(self, one_batch, sess):
        loss, pred = sess.run([self.loss, self.y_pred], feed_dict = {self.X : one_batch[0],
                                                                     self.Q : one_batch[1],
                                                                     self.Y : one_batch[2],
                                                                     self.input_keep_prob : 1.0,
                                                                     self.output_keep_prob : 1.0})
        return loss, pred