from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import pathlib
from utils.utils import rmse_loss

from models.GAIN_sklearn import GAIN_sklearn
from models.MIDA_sklearn import MIDA_sklearn

import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Load data
    file_name = '../data/house_recoded.csv'
    model_name = "gain"
    save_name = "house_modelchecking"
    miss_mechanism = "MCAR"
    data_df = pd.read_csv(file_name)
    data_x = data_df.values.astype(np.float32)

    num_index = list(range(-8, 0))
    cat_index = list(range(-data_df.shape[1], -8))

    # get all possible levels for categorical variable
    all_levels = [np.unique(x) for x in data_x[:, cat_index].T]
    all_levels_dict = dict(zip(data_df.columns[cat_index], all_levels))

    num_samples = 10
    num_imputations = 10

    i = 0
    file_name = '../samples/{}/{}/sample_{}.csv'.format(save_name, miss_mechanism, i)
    data_x_i = np.loadtxt('../samples/{}/complete/sample_{}.csv'.format(save_name, i), delimiter=",").astype(np.float32)

    miss_data_x = np.loadtxt(file_name, delimiter=",").astype(np.float32)
    data_m = 1 - np.isnan(miss_data_x).astype(np.float32)

    no, dim = miss_data_x.shape
    shuffle = np.arange(no)
    np.random.shuffle(shuffle)

    data_x_i = data_x_i[shuffle]
    miss_data_x = miss_data_x[shuffle]
    data_m = data_m[shuffle]

    miss_data_x_train = miss_data_x[:int(0.5*no)]
    data_m_train = data_m[:int(0.5*no)]

    miss_data_x_test = miss_data_x[int(0.5 * no):]
    data_m_test = data_m[int(0.5 * no):]

    # GAIN
    batch_size = 512
    alpha = 100
    iterations = 20
    hint_rate = 0.33
    num_hidden = sum([len(x) for x in all_levels]) + len(num_index)
    gain = GAIN_sklearn(num_hidden, cat_index, num_index, all_levels, batch_size, hint_rate, alpha, iterations, 1)

    train_loss, test_loss = gain.fit(miss_data_x_train, data_x_i[:int(0.5 * no)], miss_data_x_test, data_x_i[int(0.5 * no):], True)

    # MIDA
    # num_base_nodes = num_hidden
    # learning_rate = 0.001
    # batch_size = 512
    # num_steps_phase1 = 20
    # num_steps_phase2 = 2
    # theta = 7
    # mida = MIDA_sklearn(num_base_nodes, cat_index, num_index, all_levels, learning_rate, num_steps_phase1, num_steps_phase2, batch_size, theta)
    # train_loss, test_loss = mida.fit(miss_data_x_train, data_x_i[:int(0.5 * no)], miss_data_x_test,
    #                                  data_x_i[int(0.5 * no):], True)
    plt.plot(train_loss, label="training loss")
    plt.plot(test_loss, label="validation loss")
    plt.xlabel("iterations")
    plt.legend()
    plt.show()