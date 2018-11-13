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
RMSE_STOP_THRESHOLD = 0.00000001

def get_num_params():
    num_params = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        num_params += reduce(mul, [dim.value for dim in shape], 1)
    return num_params

def display(y_real_list, y_pred_list):
    pass

def run_one_epoch(model, batch_data, y_scalers, sess, is_train = True, display = False):
    if is_train :
        run_func = model.train
    else:
        run_func = model.predict

    all_loss = []
    all_mape = []
    all_rmse = []

    # for display
    y_pred_flat = []
    y_real_flat = []

    for ds in batch_data:
        loss, pred = run_func(ds, sess)

        # # un-norm the real value
        y_pre_list = []
        y_real_list = []

        if model.config.is_scaled:
            for j in range(len(pred)):
                for k in range(len(pred[j])):
                    y_pre_list.append(y_scalers[k].inverse_transform([[ pred[j][k] ]]))
                    y_real_list.append(y_scalers[k].inverse_transform([[ ds[2][j][k] ]]))
        else:
            y_pre_list = pred
            y_real_list = ds[2]

        if display:
            y_pred_flat.append(y_pre_list)
            y_real_flat.append(y_real_list)


        #mape = np.mean( np.divide(abs(np.subtract(y_pre_list, y_real_list)), y_real_list))
        mape = 0
        rmse = np.sqrt(np.mean(np.subtract(y_pre_list, y_real_list) ** 2))
        all_mape.append(mape)
        all_rmse.append(rmse)

        all_loss.append(loss)

    if display:
        y_pred_flat = np.reshape(y_pred_flat, [-1])
        y_real_flat = np.reshape(y_real_flat, [-1])
        plt.plot(y_pred_flat, 'b')
        plt.plot(y_real_flat, 'r')
        plt.show()

    return np.mean(all_loss), np.mean(all_mape), np.mean(all_rmse)

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
    dataset, scalers = ds_handler.get_dataset()
    #y_index = -55
    #y_scaler = scalers[y_index]

    train_ds, valid_ds = ds_handler.divide_two_ds(dataset)
    print('Train ds:', train_ds.shape)
    print('Valid ds:', valid_ds.shape)

    train_batch_data = ds_handler.get_all_batch_data(train_ds)
    valid_batch_data = ds_handler.get_all_batch_data(valid_ds)
    # run model
    if is_train:
        sess.run(tf.global_variables_initializer())
        print('Trainable parameter count:', get_num_params())
        print('The model saved path:', model_path)
        last_loss = 100.0
        best_valid_rmse = (100.0, 0)
        epochs = 1000

        print('Start training...')
        for i in range(epochs):
            start_t = time.time()
            #train_batch_data = np.random.shuffle(train_batch_data)
            loss, mape, rmse = run_one_epoch(model, train_batch_data, scalers, sess, True)
            print('Epoch', i, 'Train Loss:', loss, 'MAPE(%):', mape * 100, 'RMSE:', rmse, 'Cost time(min):', (time.time() - start_t) / 60)
            if abs(last_loss - loss) < RMSE_STOP_THRESHOLD:
                break
            last_loss = loss

            if i % 10 == 0:
                loss, mape, rmse = run_one_epoch(model, valid_batch_data, scalers, sess, False, display = False)
                print('Valid Loss:', loss, 'MAPE(%):', mape * 100, 'RMSE:', rmse)
                if best_valid_rmse[0] > rmse:
                    best_valid_rmse = (rmse, i)
                    # save model
                    saver.save(sess, model_path)
    else:
        saver.restore(sess, model_path)
        loss, mape, rmse = run_one_epoch(model, valid_batch_data, scalers, sess, False)
        print('Valid Loss:', loss, 'MAPE(%):', mape * 100, 'RMSE:', rmse)  
