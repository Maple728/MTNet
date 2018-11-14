class NasdaqConfig:

    T = 12 # timestep
    W = 3 # convolution window size (convolution filter height)
    n = 6 # the number of the long-term memory series
    highway_window = 6 # the window size of ar model

    D = 82  # input's variable dimension (convolution filter width)
    K = 1 # output's variable dimension
    horizon = 1 # the horizon of predicted value

    en_conv_hidden_size = 64
    en_rnn_hidden_sizes = [64]

    input_keep_prob = 1.0
    output_keep_prob = 1.0

    lr = 0.001

    batch_size = 100

    is_scaled = True
    feature_range = (0, 1)

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

    is_scaled = True
    feature_range = (0, 1)

class BJpmConfig:

    T = 12 # timestep
    W = 3 # convolution window size (convolution filter height)
    n = 7 # the number of the long-term memory series
    highway_window = 6  # the window size of ar model

    D = 7  # input's variable dimension (convolution filter width)
    K = 1 # output's variable dimension

    horizon = 3 # the horizon of predicted value

    en_conv_hidden_size = 64
    en_rnn_hidden_sizes = [64]  # last size is equal to en_conv_hidden_size

    input_keep_prob = 1.0
    output_keep_prob = 1.0

    is_scaled = True
    feature_range = (0, 1)

    lr = 0.001
    batch_size = 100
