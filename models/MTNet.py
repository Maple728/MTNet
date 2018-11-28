import tensorflow as tf
class MTNet:
    def __init__(self, config, scope = 'MTNet'):
        self.config = config
        with tf.variable_scope(scope, reuse = False):
            X = tf.placeholder(tf.float32, shape = [None, self.config.n, self.config.T, self.config.D])
            Q = tf.placeholder(tf.float32, shape = [None, self.config.T, self.config.D])
            Y = tf.placeholder(tf.float32, shape = [None, self.config.K])

            lr = tf.placeholder(tf.float32)
            input_keep_prob = tf.placeholder(tf.float32)
            output_keep_prob = tf.placeholder(tf.float32)

            # ------- no-linear component----------------
            last_rnn_hid_size = self.config.en_rnn_hidden_sizes[-1]
            # <batch_size, n, en_rnn_hidden_sizes>
            m_is = self.__encoder(X, self.config.n, scope = 'm')
            c_is = self.__encoder(X, self.config.n, scope = 'c')
            # <batch_size, 1, en_rnn_hidden_sizes>
            u = self.__encoder(tf.reshape(Q, shape = [-1, 1, self.config.T, self.config.D]), 1, scope = 'in')

            p_is = tf.matmul(m_is, tf.transpose(u, perm = [0, 2, 1]))

            # using softmax
            p_is = tf.squeeze(p_is, axis = [-1])
            p_is = tf.nn.softmax(p_is)
            # <batch_size, n, 1>
            p_is = tf.expand_dims(p_is, -1)

            # using sigmoid
            # p_is = tf.nn.sigmoid(p_is)

            # for summary
            #p_is_mean, _ = tf.metrics.mean_tensor(p_is, updates_collections = 'summary_ops', name = 'p_is')
            #tf.summary.histogram('p_is', p_is_mean)

            # <batch_size, n, en_rnn_hidden_sizes> = <batch_size, n, en_rnn_hidden_sizes> * <batch_size, n, 1>
            o_is = tf.multiply(c_is, p_is)

            pred_w = tf.get_variable('pred_w', shape = [last_rnn_hid_size * (self.config.n + 1), self.config.K],
                                     dtype = tf.float32, initializer = tf.truncated_normal_initializer(stddev = 0.1))
            pred_b = tf.get_variable('pred_b', shape = [self.config.K],
                                     dtype = tf.float32, initializer = tf.constant_initializer(0.1))
            
            pred_x = tf.concat([o_is, u], axis = 1)
            pred_x = tf.reshape(pred_x, shape = [-1, last_rnn_hid_size * (self.config.n + 1)])

            # <batch_size, D>
            y_pred = tf.matmul(pred_x, pred_w) + pred_b

            # ------------ ar component ------------
            with tf.variable_scope('AutoRegression'):
                if self.config.highway_window > 0:
                    highway_ws = tf.get_variable('highway_ws', shape = [self.config.highway_window * self.config.D, self.config.K],
                                                dtype = tf.float32,
                                                initializer = tf.truncated_normal_initializer(stddev = 0.1))
                    highway_b = tf.get_variable('highway_b', shape = [self.config.K], dtype = tf.float32,
                                                initializer = tf.constant_initializer(0.1))

                    highway_x = tf.reshape(Q[:, -self.config.highway_window:], shape = [-1, self.config.highway_window * self.config.D])
                    y_pred_l = tf.matmul(highway_x, highway_ws) + highway_b

                    # y_pred_l = tf.matmul(Q[:, -1], highway_ws[0]) + highway_b
                    # _, y_pred_l = tf.while_loop(lambda i, _ : tf.less(i, self.config.highway_window),
                    #                             lambda i, acc : (i + 1, tf.matmul(Q[:, self.config.T - i - 1], highway_ws[i]) + acc),
                    #                             loop_vars = [1, y_pred_l])
                    y_pred += y_pred_l


        # metrics summary
        #mae_loss, _ = tf.metrics.mean_absolute_error(Y, y_pred, updates_collections = 'summary_ops', name = 'mae_metric')
        #tf.summary.scalar('mae_loss', mae_loss)

        rmse_loss, _ = tf.metrics.root_mean_squared_error(Y, y_pred, updates_collections = 'summary_ops', name = 'rmse_metric')
        tf.summary.scalar("rmse_loss", rmse_loss)

        statistics_vars = tf.get_collection(tf.GraphKeys.METRIC_VARIABLES)
        statistics_vars_initializer = tf.variables_initializer(var_list = statistics_vars)

        loss = tf.losses.absolute_difference(Y, y_pred)
        with tf.name_scope('Train'):
            train_op = tf.train.AdamOptimizer(lr).minimize(loss)

        # assignment
        self.X = X
        self.Q = Q
        self.Y = Y
        self.input_keep_prob = input_keep_prob
        self.output_keep_prob = output_keep_prob
        self.lr = lr
        self.y_pred = y_pred
        self.loss = loss
        self.train_op = train_op

        self.reset_statistics_vars = statistics_vars_initializer
        self.merged_summary = tf.summary.merge_all()
        self.summary_updates = tf.get_collection('summary_ops')

    def __encoder(self, origin_input_x, n, strides = [1, 1, 1, 1], padding = 'VALID', activation_func = tf.nn.relu, scope = 'Encoder'):
        '''
            Treat batch_size dimension and n dimension as one batch_size dimension (batch_size * n).
        :param input_x:  <batch_size, n, T, D>
        :param strides:
        :param padding:
        :param scope:
        :return: the embedded of the input_x <batch_size, n, last_rnn_hid_size>
        '''
        # constant
        scope = 'Encoder_' + scope
        batch_size_new = self.config.batch_size * n
        Tc = self.config.T - self.config.W + 1
        last_rnn_hidden_size = self.config.en_rnn_hidden_sizes[-1]

        # reshape input_x : <batch_size * n, T, D, 1>
        input_x = tf.reshape(origin_input_x, shape = [-1, self.config.T, self.config.D, 1])

        with tf.variable_scope(scope, reuse = tf.AUTO_REUSE):
            # cnn parameters
            with tf.variable_scope('CNN', reuse = tf.AUTO_REUSE):
                w_conv1 = tf.get_variable('w_conv1', shape = [self.config.W, self.config.D, 1, self.config.en_conv_hidden_size], dtype = tf.float32, initializer = tf.truncated_normal_initializer(stddev = 0.1))
                b_conv1 = tf.get_variable('b_conv1', shape = [self.config.en_conv_hidden_size], dtype = tf.float32, initializer = tf.constant_initializer(0.1))

                # <batch_size_new, Tc, 1, en_conv_hidden_size>
                h_conv1 = activation_func(tf.nn.conv2d(input_x, w_conv1, strides, padding = padding) + b_conv1)
                if self.config.input_keep_prob < 1:
                    h_conv1 = tf.nn.dropout(h_conv1, self.config.input_keep_prob)


            # tmporal attention layer and gru layer
            # rnns
            rnns = [tf.nn.rnn_cell.GRUCell(h_size, activation = activation_func) for h_size in self.config.en_rnn_hidden_sizes]
            # dropout
            if self.config.input_keep_prob < 1 or self.config.output_keep_prob < 1:
                rnns = [tf.nn.rnn_cell.DropoutWrapper(rnn,
                                                      input_keep_prob = self.config.input_keep_prob,
                                                      output_keep_prob = self.config.output_keep_prob)
                        for rnn in rnns]

            if len(rnns) > 1:
                rnns = tf.nn.rnn_cell.MultiRNNCell(rnns)
            else:
                rnns = rnns[0]

            # attention layer

            # attention weights
            attr_v = tf.get_variable('attr_v', shape = [Tc, 1], dtype = tf.float32, initializer = tf.truncated_normal_initializer(stddev = 0.1))
            attr_w = tf.get_variable('attr_w', shape = [last_rnn_hidden_size, Tc], dtype = tf.float32, initializer = tf.truncated_normal_initializer(stddev = 0.1))
            attr_u = tf.get_variable('attr_u', shape = [Tc, Tc], dtype = tf.float32, initializer = tf.truncated_normal_initializer(stddev = 0.1))

            # rnn inputs
            # <batch_size, n, Tc, en_conv_hidden_size>
            rnn_input = tf.reshape(h_conv1, shape=[-1, n, Tc, self.config.en_conv_hidden_size])

            # n * <batch_size, last_rnns_size>
            res_hstates = tf.TensorArray(tf.float32, n)
            for k in range(n):
                # <batch_size, en_conv_hidden_size, Tc>
                attr_input = tf.transpose(rnn_input[:, k], perm = [0, 2, 1])

                # <batch_size, last_rnn_hidden_size>
                s_state = rnns.zero_state(self.config.batch_size, tf.float32)
                if len(self.config.en_rnn_hidden_sizes) > 1:
                    h_state = s_state[-1]
                else:
                    h_state = s_state

                for t in range(Tc):
                    # attr_v = tf.Variable(tf.truncated_normal(shape=[Tc, 1], stddev=0.1, dtype=tf.float32), name='attr_v')
                    # attr_w = tf.Variable(tf.truncated_normal(shape=[last_rnn_hidden_size, Tc], stddev=0.1, dtype=tf.float32), name='attr_w')
                    # attr_u = tf.Variable(tf.truncated_normal(shape=[Tc, Tc], stddev=0.1, dtype=tf.float32), name='attr_u')

                    # h(t-1) dot attr_w
                    h_part = tf.matmul(h_state, attr_w)

                    # en_conv_hidden_size * <batch_size_new, 1>
                    e_ks = tf.TensorArray(tf.float32, self.config.en_conv_hidden_size)
                    _, output = tf.while_loop(lambda i, _ : tf.less(i, self.config.en_conv_hidden_size),
                                              lambda i, output_ta : (i + 1, output_ta.write(i, tf.matmul(tf.tanh( h_part + tf.matmul(attr_input[:, i], attr_u) ), attr_v))),
                                              [0, e_ks])
                    # <batch_size, en_conv_hidden_size, 1>
                    e_ks = tf.transpose(output.stack(), perm = [1, 0, 2])
                    e_ks = tf.reshape(e_ks, shape = [-1, self.config.en_conv_hidden_size])

                    # <batch_size, en_conv_hidden_size>
                    a_ks = tf.nn.softmax(e_ks)

                    x_t = tf.matmul( tf.expand_dims(attr_input[:, :, t], -2), tf.matrix_diag(a_ks))
                    # <batch_size, en_conv_hidden_size>
                    x_t = tf.reshape(x_t, shape = [-1, self.config.en_conv_hidden_size])

                    h_state, s_state = rnns(x_t, s_state)

                res_hstates = res_hstates.write(k, h_state)

        return tf.transpose(res_hstates.stack(), perm = [1, 0, 2])

    def train(self, one_batch, sess):
        fd = self.get_feed_dict(one_batch, True)
        _, loss, pred = sess.run([self.train_op, self.loss, self.y_pred], feed_dict = fd)
        sess.run(self.summary_updates, feed_dict = fd)
        return loss, pred

    def predict(self, one_batch, sess):
        fd = self.get_feed_dict(one_batch, False)
        loss, pred = sess.run([self.loss, self.y_pred], feed_dict = fd)
        sess.run(self.summary_updates, feed_dict = fd)
        return loss, pred

    def get_feed_dict(self, one_batch, is_train):
        if is_train:
            fd = {self.X : one_batch[0],
                  self.Q : one_batch[1],
                  self.Y : one_batch[2],
                  self.input_keep_prob : self.config.input_keep_prob,
                  self.output_keep_prob : self.config.output_keep_prob,
                  self.lr : self.config.lr}
        else:
            fd = {self.X : one_batch[0],
                  self.Q : one_batch[1],
                  self.Y : one_batch[2],
                  self.input_keep_prob : 1.0,
                  self.output_keep_prob :1.0,
                  self.lr: self.config.lr}
        return fd;