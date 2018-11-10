class NasdaqConfig:

    T = 12 # timestep
    D = 1 # input's variable dimension (convolution filter width)
    W = 3 # convolution window size (convolution filter height)
    n = 7 # the number of the long-term memory series
    highway_window = 6 # the window size of ar model
    horizon = 1 # the horizon of predicted value

    en_conv_hidden_size = 64
    en_rnn_hidden_sizes = [64]

    input_keep_prob = 1.0
    output_keep_prob = 1.0

    lr = 0.001

    batch_size = 100

    is_scaled = True
    feature_range = (0, 1)