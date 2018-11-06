from models.MTNet import *
from configs.config import *
from preprocess.get_data import *

import numpy as np
import tensorflow as tf

is_train = True
model_path = './checkpoints/mtnet.ckpt'
RMSE_STOP_THRESHOLD = 0.0001

config = NasdaqConfig
ds_handler = NasdaqDataset(config)

def run_one_epoch(model, batch_data, y_scaler, sess, is_train = True):
    if is_train :
        run_func = model.train
    else:
        run_func = model.predict

    all_loss = []
    all_mape = []
    all_rmse = []    
    for ds in batch_data:
        loss, pred = run_func(ds, sess)
        if not is_train:
            # un-norm the real value
            y_pre_list = []
            y_real_list = []

            for j in range(len(ds[-1])):
                if model.config.is_scaled:
                    y_pre_list.append(y_scaler.inverse_transform([ pred[j] ]))
                    y_real_list.append(y_scaler.inverse_transform([ ds[2][j] ]))

            mape = np.mean( np.divide(abs(np.subtract(y_pre_list, y_real_list)), y_real_list))
            rmse = np.sqrt(np.mean(np.subtract(y_pre_list, y_real_list) ** 2))

            all_mape.append(mape)
            all_rmse.append(rmse)

        all_loss.append(loss)
    return np.mean(all_loss), np.mean(all_mape), np.mean(all_rmse)
    

# ---------
if __name__ == '__main__':

    # build model
    sess = tf.Session()
    model = MTNet(config)
    saver = tf.train.Saver()

    # tensor board
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('graphs', sess.graph)

    # data process
    dataset = ds_handler.get_dataset()
    y_scaler = dataset[-1, -1, 0]
    end_index = int((len(dataset[-1]) - 1) * 0.9)
    train_ds = dataset[-1, : end_index]
    valid_ds = dataset[-1, end_index - (1 + config.n) * config.T : -1]

    # run model
    if is_train:
        sess.run(tf.global_variables_initializer())

        last_loss = 100.0
        best_valid_res = (100.0, 0)
        epochs = 1000
        print('Start training...')
        for i in range(epochs):
            batch_data = ds_handler.get_batch_data(train_ds)
            loss, _, _ = run_one_epoch(model, batch_data, y_scaler, sess, True)
            print('Epoch', i, 'Train Loss:', loss)
            if abs(last_loss - loss) < RMSE_STOP_THRESHOLD:
                break
            last_loss = loss

            if i % 10 == 0:
                batch_data = ds_handler.get_batch_data(valid_ds)
                loss, mape, rmse = run_one_epoch(model, batch_data, y_scaler, sess, False)
                print('Valid Loss:', loss, 'MAPE(%):', mape * 100, 'RMSE:', rmse)
                if best_valid_rmse[0] > rmse:
                    best_valid_rmse = (rmse, i)
                    # save model
                    saver.save(sess, model_path)
    else:
        saver.restore(sess, model_path)
        batch_data = ds_handler.get_batch_data(valid_ds)
        loss, mape, rmse = run_one_epoch(model, batch_data, y_scaler, sess, False)
        print('Valid Loss:', loss, 'MAPE(%):', mape * 100, 'RMSE:', rmse)  
