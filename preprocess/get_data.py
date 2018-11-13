import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class Dataset(object):

    def __init__(self, config):
        self.config = config

    def get_batch_data(self, data, y_index = None):
        '''
        Get X <batch_size, n, T, D>, Q <batch_size, T, D>， Y <batch_size, K>
        :param data: time series data <length, D>
        :param y_index: Negative The y's index of config.D. if is not None, then the last K elements in D.
        :return: X <batch_size, n, T, D>, Q <batch_size, T, D>， Y <batch_size, K>
        '''
        config = self.config
        type_size = data.shape[1]

        batch_data = []
        one_batch_input_len = config.T * (config.n + 1)
        for i in range(one_batch_input_len, data.shape[0] - config.horizon + 1):
            a_batch = np.append(data[i - one_batch_input_len : i, :], data[i + config.horizon - 1, :])
            batch_data.append(a_batch)

            if len(batch_data) == config.batch_size:
                batch_data = np.array(batch_data)
                x = np.reshape( batch_data[:, : config.T * config.n * type_size], [-1, config.n, config.T, config.D])
                q = np.reshape(batch_data[:, (one_batch_input_len - config.T) * type_size: one_batch_input_len * type_size],
                               [-1, config.T, config.D])
                if y_index is None:
                    y = np.reshape(batch_data[:, -config.K : ], [-1, config.K])
                else:
                    y = np.reshape(batch_data[:, y_index], [-1, config.K])
                yield x, q, y

                batch_data = []

    def get_all_batch_data(self, data, y_index = None):
        all_batch_data = []
        batch_data = self.get_batch_data(data, y_index)
        for ds in batch_data:
            all_batch_data.append(ds)

        return all_batch_data

    def divide_three_ds(self, dataset):
        '''
        :param dataset: <length, D>
        :return: Three dataset.
        '''
        config = self.config
        train_end_index = int(dataset.shape[0] / 10 * 8)
        valid_end_index = int(dataset.shape[0] / 10 * 9)

        records_offset = (config.n + 1) * config.T

        return dataset[: train_end_index], dataset[train_end_index -  records_offset: valid_end_index], dataset[valid_end_index - records_offset :]

    def divide_two_ds(self, dataset):
        '''
        :param dataset: <length, D>
        :return: Three dataset.
        '''
        config = self.config
        train_end_index = int(dataset.shape[0] / 10 * 9)

        records_offset = (config.n + 1) * config.T

        return dataset[: train_end_index], dataset[train_end_index -  records_offset: ]

class NasdaqDataset(Dataset):
    name = 'Nasdaq'
    data_filename = './datasets/nasdaq100_padding.csv'
    
    config = None
    def __init__(self, config):
        Dataset.__init__(self, config)

    def get_dataset(self):
        '''
        Get dataset <length, D>
        :return: a tuple like (<stock_size, length, 1>, <scaler_size>) if isScaled is true, otherwise (<stock_size, length, 1>, None)
        '''
        config = self.config
        tmp_df = pd.read_csv(self.data_filename)
        data_list = np.array(tmp_df)
        # <stock, length>
        scalers = []
        data = data_list.transpose()
        data = data.reshape((data.shape[0], data.shape[1], 1))

        if config.is_scaled:
            data = data.tolist()
            for i in range(len(data)):
                scaler = MinMaxScaler( config.feature_range )
                data[i] = scaler.fit_transform(data[i]).tolist()
                scalers.append(scaler)

        # transoform <type_size, length, 1> to <length, type_size>
        data = np.transpose(np.squeeze(data, axis = -1))
        return data, scalers

class TaxiNYDataset(Dataset):
    time_interval = 30
    name = 'TaxiNY_%s' %  time_interval
    data_filename = './datasets/grid_map_dict_%smin.pickle' % time_interval

    def __init__(self, config):
        Dataset.__init__(self, config)

    def get_dataset(self):
        '''
        Get dataset <length, D>
        :return: a tuple like (<length, D>, <scaler_size(D)>) if isScaled is true, otherwise (<length, D>, [])
        '''
        config = self.config
        # get data and process shape
        # <n, lat_len, lon_len>
        with open(self.data_filename, 'rb') as f:
            grid_map_dict = pickle.load(f)

        # transform dict to grid map list ordered by key ascending
        # sort asc
        grid_map_list = sorted(grid_map_dict.items(), key = lambda item : item[0])
        grid_map_list = list(map(lambda item : item[1], grid_map_list))
        
        # transform shape to <lat_len * lon_len, time dim, 1>
        data = np.reshape(grid_map_list, (len(grid_map_list), -1, 1) )
        data = data.transpose( (1, 0, 2))

        scalers = []

        if config.is_scaled:
            data = data.tolist()
            for i in range(len(data)):
                scaler = MinMaxScaler( config.feature_range )
                data[i] = scaler.fit_transform(data[i]).tolist()
                scalers.append(scaler)

        # transoform <type_size, length, 1> to <length, type_size>
        data = np.transpose(np.squeeze(data, axis=-1))
        return np.array(data), scalers

class BJPMDataset(Dataset):
    name = 'BJPM2_5'
    data_filename = './datasets/grid_map_dict_%smin.pickle'

    def __init__(self, config):
        Dataset.__init__(self, config)


    def get_dataset(self):
        '''
        Get dataset <length, D>
        :return: a tuple like (<length, D>, <scaler_size(D)>) if isScaled is true, otherwise (<length, D>, [])
        '''
        pass

