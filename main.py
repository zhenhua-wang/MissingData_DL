from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from DAE_imputation_v2 import autoencoder_imputation
from gain_categorical_v2 import gain

if __name__ == '__main__':
    miss_rate = 0.3
    num_samples = 200

    DAE_parameters = {'learning_rate': 0.001,
                        'batch_size': 256,
                        'num_steps_phase1': 100,
                        'num_steps_phase2': 3,
                        'theta': 16}

    gain_parameters = {'batch_size': 512,
                       'hint_rate': 0.15,
                       'alpha': 5,
                       'iterations': 100,
                       'h_Gdim': 32,
                       'h_Ddim': 32}

    # Load data
    file_name = 'data/house.csv'
    house_df = pd.read_csv(file_name)
    data_x = house_df.values.astype(np.float32)

    num_index = list(range(-8, 0))
    cat_index = list(range(-house_df.shape[1], -8))

    # get all possible levels for categorical variable
    all_levels = [np.unique(x) for x in data_x[:, cat_index].T]
    all_levels_dict = dict(zip(house_df.columns[cat_index], all_levels))

    for i in range(num_samples):
        file_name = './samples/MCAR/sample_{}.csv'.format(i)
        miss_data_x = np.loadtxt(file_name, delimiter=",").astype(np.float32)
        data_m = 1-np.isnan(miss_data_x).astype(np.float32)

        # imputed_data_x, loss_list = autoencoder_imputation(miss_data_x, data_m,
        #                                                    cat_index, num_index,
        #                                                    all_levels, DAE_parameters)
        imputed_data_x, Gloss_list, Dloss_list = gain(miss_data_x, data_m,
                                                      cat_index, num_index,
                                                      all_levels, gain_parameters)
        plt.plot(Gloss_list)
        plt.plot(Dloss_list)
        plt.show()
        np.savetxt("./results/GAIN/imputed_{}.csv".format(i), imputed_data_x, delimiter=",")