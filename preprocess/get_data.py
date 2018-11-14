import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class Dataset(object):
    '''
    Subclass must implements below functions:
        get_dataset()
        scaling(data)
    Custom overwrited functions:
        get_batch_data(ds_type)
        inverse_transform(y_scaled)
    '''

    def __init__(self, config):
        self.config = config

        config = self.config
        dataset = self.get_dataset()
        if config.is_scaled:
            self.dataset, self.scaler = self.scaling(dataset)
        else:
            self.dataset, self.scaler = dataset, None

        self.train_ds, self.valid_ds = self.divide_two_ds(self.dataset)
        print('-Train dataset shape:', self.train_ds.shape)
        print('-Valid dataset shape:', self.valid_ds.shape)

    def get_batch_data(self, ds_type):
        '''
        General function, for D == K
        Get X <batch_size, n, T, D>, Q <batch_size, T, D>， Y <batch_size, D>
        :param ds_type: A string. 'T' : train ds; 'V' : valid ds; 'E' : test ds.
        :return: X <batch_size, n, T, D>, Q <batch_size, T, D>， Y <batch_size, D>
        '''

        config = self.config
        if ds_type == 'T':
            ds = self.train_ds
        elif ds_type == 'V':
            ds = self.valid_ds
        elif ds_type == 'E':
            ds = self.test_ds
        else:
            raise RuntimeError("Unknown dataset type['T','V', 'E']:", ds_type)

        batch_data = []
        one_batch_x_len = config.T * (config.n + 1)
        for i in range(one_batch_x_len, ds.shape[0] - config.horizon + 1):
            a_batch = np.append(ds[i - one_batch_x_len : i, :], ds[i + config.horizon - 1, :])
            batch_data.append(a_batch)

            if len(batch_data) == config.batch_size:
                batch_data = np.array(batch_data)
                x_endindex = config.T * config.n * config.D
                x = np.reshape( batch_data[:, : x_endindex], [-1, config.n, config.T, config.D])
                q = np.reshape(batch_data[:, x_endindex : -config.D],
                               [-1, config.T, config.D])
                y = np.reshape(batch_data[:, -config.D :], [-1, config.D])
                yield x, q, y

                batch_data = []

    def get_all_batch_data(self, ds_type = 'T'):
        all_batch_data = []
        batch_data = self.get_batch_data(ds_type)
        for ds in batch_data:
            all_batch_data.append(ds)

        return all_batch_data

    def inverse_transform(self, y_scaled):
        '''
        General function, for K == D
        :param y_scaled: <samples, features>
        :return: real y
        '''
        y_scaler = self.scaler

        y_scaled = np.reshape(y_scaled, [-1, self.config.K])
        real_y = y_scaler.inverse_transform(y_scaled)

        return real_y

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
        :return: Two dataset.
        '''
        config = self.config
        train_end_index = int(dataset.shape[0] / 10 * 9)

        records_offset = (config.n + 1) * config.T

        return dataset[: train_end_index], dataset[train_end_index -  records_offset: ]

class TaxiNYDataset(Dataset):
    time_interval = 30
    name = 'TaxiNY_%s' %  time_interval
    data_filename = './datasets/grid_map_dict_%smin.pickle' % time_interval

    def __init__(self, config):
        Dataset.__init__(self, config)

    def get_dataset(self):
        '''
        Get dataset <length, D>
        :return: <length, D>
        '''

        config = self.config
        # get data and process shape
        # <n, lat_len, lon_len>
        with open(self.data_filename, 'rb') as f:
            grid_map_dict = pickle.load(f)
        # transform dict to grid_map list ordered by key ascending
        # sort asc
        grid_map_list = sorted(grid_map_dict.items(), key = lambda item : item[0])
        grid_map_list = list(map(lambda item : item[1], grid_map_list))
        # <lat_len * lon_len, time dim>
        return np.reshape(grid_map_list, [len(grid_map_list), -1])

    def scaling(self, data):
        '''
        :param data: <length, columns>
        :return: a tuple (scaled_data<same shape as input>, scaler)
        '''
        config = self.config
        scaler = MinMaxScaler(config.feature_range)
        data = scaler.fit_transform(data.tolist())

        return data, scaler

class BJPMDataset(Dataset):
    name = 'BJPM2_5'
    data_filename = './datasets/PRSA_data_7col.csv'

    def __init__(self, config):
        Dataset.__init__(self, config)

    def get_dataset(self):
        '''
        Get dataset <length, D + 1>
        :return: <length, D + 1>, the last column is y
        '''
        config = self.config
        # the last element in row is pm2.5
        records = pd.read_csv(self.data_filename)
        data = np.array(records)

        # <length, columns>
        return np.transpose(data)

    def scaling(self, data):
        '''
        :param data: <length, columns>
        :return: a tuple (scaled_data<same shape as input>, scalers<columns>)
        '''
        config = self.config

        # <columns, length, 1>
        data = np.expand_dims(data, axis = -1)
        scalers = []

        if config.is_scaled:
            data = data.tolist()
            for i in range(len(data)):
                scaler = MinMaxScaler( config.feature_range )
                data[i] = scaler.fit_transform(data[i]).tolist()
                scalers.append(scaler)

        # <length, columns>
        data = np.transpose(np.squeeze(data, axis=-1))
        return np.array(data), scalers

    def get_batch_data(self, ds_type):
        '''
        Get X <batch_size, n, T, D>, Q <batch_size, T, D>， Y <batch_size, K>
        :param ds_type: A string. 'T' : train ds; 'V' : valid ds; 'E' : test ds.
        :return: X <batch_size, n, T, D>, Q <batch_size, T, D>， Y <batch_size, K>
        '''
        if ds_type == 'T':
            ds = self.train_ds
        elif ds_type == 'V':
            ds = self.valid_ds
        elif ds_type == 'E':
            ds = self.test_ds
        else:
            raise RuntimeError("Unknown dataset type['T','V', 'E']:", ds_type)

        config = self.config
        x_data = ds[:, : -1]
        y_data = ds[:, -1]

        batch_data = []
        one_batch_x_len = config.T * (config.n + 1)
        for i in range(one_batch_x_len, ds.shape[0] - config.horizon + 1):
            a_batch = np.append(x_data[i - one_batch_x_len : i, :], y_data[i + config.horizon - 1])
            batch_data.append(a_batch)

            if len(batch_data) == config.batch_size:
                batch_data = np.array(batch_data)
                x_endindex = config.T * config.n * config.D
                x = np.reshape( batch_data[:, : x_endindex], [-1, config.n, config.T, config.D])
                q = np.reshape(batch_data[:, x_endindex : -1],
                               [-1, config.T, config.D])
                y = np.reshape(batch_data[:, -1], [-1, 1])
                yield x, q, y

                batch_data = []

    def inverse_transform(self, y_scaled):
        '''
        :param y_scaled: <samples, features>
        :return: real y
        '''
        y_scaler = self.scaler[-1]
        origin_shape = [y_scaled.shape[0], y_scaled.shape[1]]
        y_scaled = np.reshape(y_scaled, [-1, 1])
        real_y = y_scaler.inverse_transform(y_scaled)

        return np.reshape(real_y, origin_shape)

