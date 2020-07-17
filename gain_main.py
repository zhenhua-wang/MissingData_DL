from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np

from gain_categorical_v2 import gain
from utils import rmse_loss, binary_sampler, onehot, reverse_onehot
from sklearn.preprocessing import OneHotEncoder

if __name__ == '__main__':
    data_name = "big_samples_releveled"
    data_miss_name = "big_samples_releveled_miss"

    gain_parameters = {'batch_size': 256,
                       'hint_rate': 0.8,
                       'alpha': 150,
                       'iterations': 1000}
    # gain_parameters = {'batch_size': 256,
    #                    'hint_rate': 0.9,
    #                    'alpha': 100,
    #                    'iterations': 500}

    # Load data
    file_name = 'data/' + data_name + '.csv'
    data_x = np.loadtxt(file_name, delimiter=",", skiprows=1).astype(np.float32)

    # Parameters
    no, dim = data_x.shape

    # Introduce missing data
    # miss_name = 'data/' + data_miss_name + '.csv'
    # data_m = np.loadtxt(miss_name, delimiter=",", skiprows=1)
    data_m = binary_sampler(1 - 0.3, no, dim)
    miss_data_x = data_x.copy()
    miss_data_x[data_m == 0] = np.nan

    # one-hot encoding
    enc = OneHotEncoder(sparse=False, dtype=np.float32)
    data_x_enc = enc.fit_transform(data_x)
    n_classes = list(map(lambda x: len(x), enc.categories_))

    # introduce missing to onehot encoded data
    data_m_enc = np.ones((no, sum(n_classes)), dtype=np.float32)
    col_index = 0
    for j in range(data_x.shape[1]):
        data_m_enc[:, col_index:col_index + n_classes[j]] = np.repeat(data_m[:, j].reshape([-1, 1]), n_classes[j],
                                                                      axis=1)
        col_index = col_index + n_classes[j]

    miss_data_x_enc = data_x_enc.copy()
    miss_data_x_enc[data_m_enc == 0] = np.nan

    # Impute missing data
    acc = []
    acc_0 = []
    acc_non0 = []
    for i in range(10):
        imputed_data_x = gain(miss_data_x_enc, data_m_enc, n_classes, gain_parameters)

        # reverse onehot encode
        imputed_data_x = enc.inverse_transform(imputed_data_x)

        # report accuracy
        impute_part = imputed_data_x[data_m == 0]
        miss_part = data_x[data_m == 0]
        num_miss = sum((data_m == 0).reshape(-1))
        acc.append(sum(impute_part == miss_part) / num_miss)
        acc_0.append(sum(impute_part[miss_part == 0] == miss_part[miss_part == 0]) / sum(data_x[data_m == 0] == 0))
        acc_non0.append(sum(impute_part[miss_part != 0] == miss_part[miss_part != 0]) / sum(data_x[data_m == 0] != 0))
    print()
    print('Accuracy Performance: mean {}, std {}'.format(np.round(np.mean(acc), 4), np.round(np.std(acc), 4)))
    print('Accuracy 0 Performance: mean {}, std {}'.format(np.round(np.mean(acc_0), 4), np.round(np.std(acc_0), 4)))
    print('Accuracy non 0 Performance: mean {}, std {}'.format(np.round(np.mean(acc_non0), 4),
                                                               np.round(np.std(acc_non0), 4)))