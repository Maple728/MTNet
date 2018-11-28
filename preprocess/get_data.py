import pickle
import numpy as np
import pandas as pd
import h5py
from sklearn.preprocessing import MinMaxScaler

class Dataset(object):
    '''
    Subclass must implements below functions:
        get_dataset()
    Custom overwrited functions:
        process2records(ds, config)
        get_batch_data()
        scaling(raw_dataset)
        inverse_transform(y_scaled)
    '''
    SACLED_FEATURE_RANGE = (0, 1)

    def __init__(self, config):
        self.config = config
        # get dataset
        self.raw_dataset = self.get_dataset()
        # get scaled dataset and sacler
        self.dataset, self.scaler = self.scaling(self.raw_dataset)

    def process2records(self, ds, config):
        '''
        :param ds: dataset <length, columns>
        :param config:
        :return: <record_nums, record_len>
        '''
        records = []
        one_record_x_len = config.T * (config.n + 1)
        for i in range(one_record_x_len, ds.shape[0] - config.horizon + 1):
            one_record = np.append(ds[i - one_record_x_len: i, :], ds[i + config.horizon - 1, :])
            records.append(one_record)

        return np.array(records)

    def get_batch_data(self, records, config):
        '''
        Get X <batch_size, n, T, D>, Q <batch_size, T, D>， Y <batch_size, K>
        :param records: <record_nums, config.D * config.T * (config.n + 2)>
        :param config:
        :return:
        '''
        i = len(records)
        one_record_len = config.D * config.T * (config.n + 1) + config.D
        while i > 0:
            start_index = i - config.batch_size
            if start_index < 0:
                break

            batch_data = records[start_index : i, :]
            x = np.reshape(batch_data[:, : config.T * config.n * config.D], [-1, config.n, config.T, config.D])
            q = np.reshape(batch_data[:, config.T * config.n * config.D : -config.D], [-1, config.T, config.D])
            y = np.reshape(batch_data[:, -config.D :], [-1, config.D])
            yield x, q, y

            i = start_index

    def scaling(self, data):
        '''
        :param data: <length, columns>
        :return: a tuple (scaled_data<same shape as input>, scaler)
        '''
        scaler = MinMaxScaler(self.SACLED_FEATURE_RANGE)
        data = scaler.fit_transform(data.tolist())

        return data, scaler

    def inverse_transform(self, y_scaled):
        '''
        General function, for K == D
        :param y_scaled: <samples, features>
        :return: real y
        '''
        y_scaler = self.scaler

        real_y = y_scaler.inverse_transform(y_scaled)

        return real_y

    def get_all_batch_data(self, config, ds_type = 'T'):
        is_shuffle = False
        if ds_type == 'T':
            ds = self.train_ds
            is_shuffle = True
        elif ds_type == 'V':
            ds = self.valid_ds
        elif ds_type == 'E':
            ds = self.test_ds
        else:
            raise RuntimeError("Unknown dataset type['T','V', 'E']:", ds_type)

        records = self.process2records(ds, config)
        if is_shuffle:
            np.random.shuffle(records)

        all_batch_data = []
        batch_data = self.get_batch_data(records, config)
        for ds in batch_data:
            all_batch_data.append(ds)

        return all_batch_data

    def divide_ds(self, config, ratios = [0.8, 0.9]):
        '''

        :param dataset: <length, D>
        :param ratios:
        :return: List of length is the length of ratios + 1
        '''

        len = self.dataset.shape[0]
        records_offset = (config.n + 1) * config.T

        ds_list = []
        prev_index = 0

        ratios.append(1.0)
        for ratio in ratios:
            cur_index = int(len * ratio)
            if prev_index != 0:
                prev_index = prev_index - records_offset

            ds_list.append( self.dataset[prev_index : cur_index])

            prev_index = cur_index

        return ds_list

class TaxiNYDataset(Dataset):
    time_interval = 30
    name = 'TaxiNY_%s' %  time_interval
    data_filename = './datasets/grid_map_dict_%smin.pickle' % time_interval

    def __init__(self, config):
        Dataset.__init__(self, config)

        self.train_ds, self.valid_ds = self.divide_ds(self.dataset, [0.9])
        print('-Train dataset shape:', self.train_ds.shape)
        print('-Valid dataset shape:', self.valid_ds.shape)

    def get_dataset(self):
        '''
        Get dataset <length, D>
        :return: <length, D>
        '''
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

class BJPMDataset(Dataset):
    name = 'BJPM2_5'
    data_filename = './datasets/PRSA_data_7col.csv'

    def __init__(self, config):
        Dataset.__init__(self, config)

        self.train_ds, self.valid_ds, self.test_ds = self.divide_ds(config, [0.6, 0.8])
        print('-Train dataset shape:', self.train_ds.shape)
        print('-Valid dataset shape:', self.valid_ds.shape)
        print('-Test dataset shape:', self.test_ds.shape)

    def get_dataset(self):
        '''
        Get dataset <length, D>
        :return: <length, D>, the last column is y
        '''
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
        # <columns, length, 1>
        data = np.expand_dims(data, axis = -1)
        scalers = []


        data = data.tolist()
        for i in range(len(data)):
            scaler = MinMaxScaler( self.SACLED_FEATURE_RANGE )
            data[i] = scaler.fit_transform(data[i]).tolist()
            scalers.append(scaler)

        # <length, columns>
        data = np.transpose(np.squeeze(data, axis=-1))
        return np.array(data), scalers

    def process2records(self, ds, config):
        '''
        :param ds: dataset <length, columns>
        :param config:
        :return: <record_nums, config.T * (config.n + 1) + 1>
        '''
        records = []

        one_record_x_len = config.T * (config.n + 1)
        for i in range(one_record_x_len, ds.shape[0] - config.horizon + 1):
            one_record = np.append(ds[i - one_record_x_len: i, :], ds[i + config.horizon - 1, -1])
            records.append(one_record)

        return np.array(records)

    def get_batch_data(self, records, config):
        '''
        Get X <batch_size, n, T, D>, Q <batch_size, T, D>， Y <batch_size, K>
        :param records: <record_nums, config.T * (config.n + 1) + 1>
        :param config:
        :return:
        '''
        i = len(records)
        while i > 0:
            start_index = i - config.batch_size
            if start_index < 0:
                break
            batch_data = records[start_index : i, :]
            x = np.reshape(batch_data[:, : config.T * config.n * config.D], [-1, config.n, config.T, config.D])
            q = np.reshape(batch_data[:, config.T * config.n * config.D : -1], [-1, config.T, config.D])
            y = np.reshape(batch_data[:, -1], [-1, 1])
            yield x, q, y

            i = start_index

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

class SolarEnergyDataset(Dataset):
    name = 'SolarEnergy_2006'
    data_filename = './datasets/solar_energy_2006_10min.csv'

    def __init__(self, config):
        Dataset.__init__(self, config)

        self.train_ds, self.valid_ds, self.test_ds = self.divide_ds(config, [0.6, 0.8])
        print('-Train dataset shape:', self.train_ds.shape)
        print('-Valid dataset shape:', self.valid_ds.shape)
        print('-Test dataset shape:', self.test_ds.shape)

    def get_dataset(self):
        '''
        Get dataset <length, D>
        :return: <length, D>
        '''

        records = pd.read_csv(self.data_filename)
        data = np.array(records)

        # <length, columns>
        return data

class BikeNYCDataset(Dataset):
    name = 'BikeNYC'
    data_filename = './datasets/NYC14_M16x8_T60_NewEnd.h5'

    def __init__(self, config):
        Dataset.__init__(self, config)

        self.train_ds, self.valid_ds = self.divide_ds(config, [0.8])
        print('-Train dataset shape:', self.train_ds.shape)
        print('-Valid dataset shape:', self.valid_ds.shape)

    def get_dataset(self):
        '''
        Get dataset <length, D>
        :return: <length, D>
        '''
        f = h5py.File(self.data_filename)
        data = np.reshape(f['data'], [f['data'].shape[0], -1])

        return data

