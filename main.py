from models.MTNet import *
from configs.config import *
from preprocess.get_data import *

from functools import reduce
from operator import mul
import time
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

CONFIG = BJpmConfig
DS_HANDLER = BJPMDataset

is_train = True

MODEL_DIR = os.path.join('logs', 'checkpoints')
LOG_DIR = os.path.join('logs', 'graphs')

def get_num_params():
    num_params = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        num_params += reduce(mul, [dim.value for dim in shape], 1)
    return num_params

def make_config_string(config):
    return "T%s_W%s_n%s_window%s_dropin%s_dropout%s_hid%s_hor%s_lr%s" % \
           (config.T, config.W, config.n, config.highway_window, config.input_keep_prob, config.output_keep_prob,
            config.en_conv_hidden_size, config.horizon, config.lr)

def make_log_dir(config):
    return os.path.join(LOG_DIR, make_config_string(config))

def make_model_path(config):
    return os.path.join(MODEL_DIR, make_config_string(config), 'mtnet.ckpt')

def run_one_epoch(sess, model, batch_data, summary_writer, ds_handler, epoch_num, is_train = True):
    # reset statistics variables
    sess.run(model.reset_statistics_vars)

    if is_train :
        run_func = model.train
    else:
        run_func = model.predict

    y_pred_list = []
    y_real_list = []
    loss_list = []

    for ds in batch_data:
        loss, pred = run_func(ds, sess)

        y_pred_list.append(pred)
        y_real_list.append(ds[-1])
        loss_list.append(loss)

    # inverse norm
    y_pred_list = np.reshape(y_pred_list, [-1, model.config.K])
    y_real_list = np.reshape(y_real_list, [-1, model.config.K])

    y_pred_list = ds_handler.inverse_transform(y_pred_list)
    y_real_list = ds_handler.inverse_transform(y_real_list)

    # real value loss
    loss = np.mean(loss_list)
    mae = np.mean(abs(np.subtract(y_pred_list, y_real_list)))
    rmse = np.sqrt(np.mean(np.subtract(y_pred_list, y_real_list) ** 2))

    # summary
    # model's summary
    summary = sess.run(model.merged_summary)
    summary_writer.add_summary(summary, epoch_num)
    # other summary
    real_mae_summary = tf.Summary()
    real_mae_summary.value.add(tag='real_mae', simple_value=mae)
    summary_writer.add_summary(real_mae_summary, epoch_num)

    real_rmse_summary = tf.Summary()
    real_rmse_summary.value.add(tag='real_rmse', simple_value=rmse)
    summary_writer.add_summary(real_rmse_summary, epoch_num)

    return loss, mae, rmse

def run_one_config(config):
    epochs = 300

    log_path = make_log_dir(config)
    model_path = make_model_path(config)
    # build model
    with tf.Session() as sess:
        model = MTNet(config)
        saver = tf.train.Saver()

        # data process
        ds_handler = DS_HANDLER(config)
        train_batch_data = ds_handler.get_all_batch_data(config, 'T')
        valid_batch_data = ds_handler.get_all_batch_data(config, 'E')

        # tensor board
        train_writer = tf.summary.FileWriter(os.path.join(log_path, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(log_path, 'test'))

        print('----------Train Config:', make_config_string(config), '. Total epochs:', epochs)
        print('Trainable parameter count:', get_num_params())

        # Actually run
        sess.run(tf.global_variables_initializer())

        best_valid_rmse = (100.0, 0)

        for i in range(epochs):
            loss, mae, rmse = run_one_epoch(sess, model, train_batch_data, train_writer, ds_handler, i, True)

            if i % 10 == 9:
                loss, mae, rmse = run_one_epoch(sess, model, valid_batch_data, test_writer, ds_handler, i, False)
                if best_valid_rmse[0] > rmse:
                    best_valid_rmse = (rmse, i)
                    # save model
                    saver.save(sess, model_path)
                    print('Epoch', i, 'Valid Loss:', loss, 'MAE:', mae, 'RMSE:', rmse)

        print('---------Best score in epoch:', best_valid_rmse[1], ' RMSE:', best_valid_rmse[0])

    # free default graph
    tf.reset_default_graph()


if __name__ == '__main__':
    config = CONFIG()
    for input_keep_prob in [0.8, 1.0]:
        config.input_keep_prob = input_keep_prob
        for highway_window in [1, 3]:
            config.highway_window = highway_window

            run_one_config(config)