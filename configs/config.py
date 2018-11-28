class TaxiNYConfig:

    T = 12 # timestep
    W = 6 # convolution window size (convolution filter height)
    n = 6 # the number of the long-term memory series
    highway_window = 12 # the window size of ar model

    D = 100  # input's variable dimension (convolution filter width)
    K = 100 # output's variable dimension
    horizon = 1 # the horizon of predicted value

    en_conv_hidden_size = 32
    en_rnn_hidden_sizes = [16, 32]  # last size is equal to en_conv_hidden_size

    input_keep_prob = 0.8
    output_keep_prob = 1.0

    lr = 0.001

    batch_size = 100

class BJpmConfig:

    def __init__(self):
        self.T = 8 # timestep
        self.W = 3 # convolution window size (convolution filter height)`
        self.n = 7 # the number of the long-term memory series
        self.highway_window = 8  # the window size of ar model

        self.D = 8  # input's variable dimension (convolution filter width)
        self.K = 1 # output's variable dimension

        self.horizon = 6 # the horizon of predicted value

        self.en_conv_hidden_size = 32
        self.en_rnn_hidden_sizes = [32]  # last size is equal to en_conv_hidden_size

        self.input_keep_prob = 0.8
        self.output_keep_prob = 1.0

        self.lr = 0.001
        self.batch_size = 100

class SolarEnergyConfig:

    def __init__(self):
        self.T = 12 # timestep
        self.W = 6 # convolution window size (convolution filter height)`
        self.n = 7 # the number of the long-term memory series
        self.highway_window = 6  # the window size of ar model

        self.D = 137  # input's variable dimension (convolution filter width)
        self.K = 137 # output's variable dimension

        self.horizon = 6 # the horizon of predicted value

        self.en_conv_hidden_size = 32
        self.en_rnn_hidden_sizes = [20, 32]  # last size is equal to en_conv_hidden_size

        self.input_keep_prob = 0.8
        self.output_keep_prob = 1.0

        self.lr = 0.003
        self.batch_size = 100

class BikeNYCConfig:

    def __init__(self):
        self.T = 24 # timestep
        self.W = 6 # convolution window size (convolution filter height)`
        self.n = 6 # the number of the long-term memory series
        self.highway_window = 12  # the window size of ar model

        self.D = 256  # input's variable dimension (convolution filter width)
        self.K = 256 # output's variable dimension

        self.horizon = 1 # the horizon of predicted value

        self.en_conv_hidden_size = 32
        self.en_rnn_hidden_sizes = [32, 32]  # last size is equal to en_conv_hidden_size

        self.input_keep_prob = 0.8
        self.output_keep_prob = 1.0

        self.lr = 0.003
        self.batch_size = 32