from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import matplotlib.pyplot as plt

from DAE_imputation_v2 import autoencoder_imputation
from utils import rmse_loss, binary_sampler
from sklearn.preprocessing import OneHotEncoder

if __name__ == '__main__':
    data_name = "big_samples_releveled"
    miss_rate = 0.3

    train_parameters = {'learning_rate': 0.01,
                        'batch_size': 128,
                        'num_steps_phase1': 3000,
                        'num_steps_phase2': 2}
    model_parameters = {'theta': 7}

    # Load data
    file_name = 'data/' + data_name + '.csv'
    data_x = np.loadtxt(file_name, delimiter=",", skiprows=1).astype(np.float32)

    # Parameters
    no, dim = data_x.shape

    # Introduce missing data
    data_m = binary_sampler(1 - miss_rate, no, dim)
    miss_data_x = data_x.copy()
    miss_data_x[data_m == 0] = np.nan

    # initial impute (most frequent elements)
    data_x_common_value = np.apply_along_axis(lambda x: np.bincount(x[~np.isnan(x)].astype(np.int)).argmax(), 0, miss_data_x)
    for j in range(miss_data_x.shape[1]):
        miss_data_x[np.isnan(miss_data_x[:, j]), j] = data_x_common_value[j]

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

    miss_data_x_enc = enc.transform(miss_data_x)


    # Impute missing data
    acc = []
    acc_0 = []
    acc_non0 = []
    for i in range(10):
        imputed_data_x, loss_list = autoencoder_imputation(miss_data_x_enc, data_m_enc, n_classes, model_parameters, train_parameters)

        # reverse onehot encode
        imputed_data_x = enc.inverse_transform(imputed_data_x)

        plt.plot(loss_list)
        plt.show()

        # report accuracy
        impute_part = imputed_data_x[data_m == 0]
        miss_part = data_x[data_m == 0]
        num_miss = sum((data_m == 0).reshape(-1))
        acc.append(sum(impute_part == miss_part) / num_miss)
        acc_0.append(sum(impute_part[miss_part == 0] == miss_part[miss_part == 0]) / sum(data_x[data_m == 0] == 0))
        acc_non0.append(sum(impute_part[miss_part != 0] == miss_part[miss_part != 0])/sum(data_x[data_m == 0] != 0))
    print()
    print('Accuracy Performance: mean {}, std {}'.format(np.round(np.mean(acc), 4), np.round(np.std(acc), 4)))
    print('Accuracy 0 Performance: mean {}, std {}'.format(np.round(np.mean(acc_0), 4), np.round(np.std(acc_0), 4)))
    print('Accuracy non 0 Performance: mean {}, std {}'.format(np.round(np.mean(acc_non0), 4), np.round(np.std(acc_non0), 4)))