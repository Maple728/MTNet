from models.MTNet import *
from configs.config import *
from preprocess.get_data import *

from functools import reduce
from operator import mul
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

config = TaxiNYConfig
ds_handler = TaxiNYDataset(config)

is_train = True
model_path = './checkpoints/%s.ckpt' % ds_handler.name

def get_num_params():
    num_params = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        num_params += reduce(mul, [dim.value for dim in shape], 1)
    return num_params

def run_one_epoch(model, batch_data, sess, is_train = True, display = False):
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
    y_pred_list = np.reshape(y_pred_list, [-1, config.K])
    y_real_list = np.reshape(y_real_list, [-1, config.K])

    y_pred_list = ds_handler.inverse_transform(y_pred_list)
    y_real_list = ds_handler.inverse_transform(y_real_list)

    if display:
        plt.plot(y_pred_list.flatten(), 'b')
        plt.plot(y_real_list.flatten(), 'r')
        plt.show()

    mae = abs(np.subtract(y_pred_list, y_real_list))
    rmse = np.sqrt(np.mean(np.subtract(y_pred_list, y_real_list) ** 2))

    return np.mean(loss_list), np.mean(mae), np.mean(rmse)

if __name__ == '__main__':

    # build model
    sess = tf.Session()
    model = MTNet(config)
    saver = tf.train.Saver()

    # tensor board
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('graphs', sess.graph)

    # data process
    print('Processing data...')

    train_batch_data = ds_handler.get_all_batch_data('T')
    valid_batch_data = ds_handler.get_all_batch_data('V')

    # run model
    if is_train:
        sess.run(tf.global_variables_initializer())
        print('Trainable parameter count:', get_num_params())
        print('The model saved path:', model_path)

        best_valid_rmse = (100.0, 0)
        epochs = 500

        print('Start training...')
        for i in range(epochs):
            start_t = time.time()
            loss, mape, rmse = run_one_epoch(model, train_batch_data, sess, True)
            print('Epoch', i, 'Train Loss:', loss, 'MAPE(%):', mape * 100, 'RMSE:', rmse, 'Cost time(min):', (time.time() - start_t) / 60)

            if i % 10 == 0:
                loss, mape, rmse = run_one_epoch(model, valid_batch_data, sess, False, display = False)
                print('Valid Loss:', loss, 'MAPE(%):', mape * 100, 'RMSE:', rmse)
                if best_valid_rmse[0] > rmse:
                    best_valid_rmse = (rmse, i)
                    # save model
                    saver.save(sess, model_path)
        print('Best score in epoch:', best_valid_rmse[1], ' RMSE:', best_valid_rmse[0])

    else:
        saver.restore(sess, model_path)
        loss, mape, rmse = run_one_epoch(model, valid_batch_data, sess, False)
        print('Valid Loss:', loss, 'MAPE(%):', mape * 100, 'RMSE:', rmse)  
