class TaxiNYConfig:

    T = 12 # timestep
    W = 3 # convolution window size (convolution filter height)
    n = 5 # the number of the long-term memory series
    highway_window = 6 # the window size of ar model

    D = 100  # input's variable dimension (convolution filter width)
    K = 100 # output's variable dimension
    horizon = 1 # the horizon of predicted value

    en_conv_hidden_size = 64
    en_rnn_hidden_sizes = [128, 64]  # last size is equal to en_conv_hidden_size

    input_keep_prob = 1.0
    output_keep_prob = 1.0

    lr = 0.001

    batch_size = 100

class BJpmConfig:

    def __init__(self):
        self.T = 8 # timestep
        self.W = 3 # convolution window size (convolution filter height)`
        self.n = 7 # the number of the long-term memory series
        self.highway_window = 3  # the window size of ar model

        self.D = 8  # input's variable dimension (convolution filter width)
        self.K = 1 # output's variable dimension

        self.horizon = 6 # the horizon of predicted value

        self.en_conv_hidden_size = 100
        self.en_rnn_hidden_sizes = [100]  # last size is equal to en_conv_hidden_size

        self.input_keep_prob = 0.8
        self.output_keep_prob = 1.0

        self.lr = 0.001
        self.batch_size = 100
