from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import pathlib
import os

from models.MIDA_v2 import autoencoder_imputation
from models.GAIN_v2 import gain
from utils.utils import rmse_loss

if __name__ == '__main__':
    num_samples = 100
    num_imputations = 10

    DAE_parameters = {'learning_rate': 0.001,
                        'batch_size': 512,
                        'num_steps_phase1': 200,
                        'num_steps_phase2': 2,
                        'theta': 7}

    gain_parameters = {'batch_size': 512,
                       'hint_rate': 0.13, # MAR
                       'alpha': 100,
                       'iterations': 200
                       }

    # Load data
    file_name = '../data/house_recoded.csv'
    model_name = "gain"
    save_name = "house"
    miss_mechanism = "MAR"
    data_df = pd.read_csv(file_name)
    data_x = data_df.values.astype(np.float32)

    save_path = "../results/{}/{}".format(save_name, miss_mechanism)
    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)

    num_index = list(range(-8, 0))
    cat_index = list(range(-data_df.shape[1], -8))

    # get all possible levels for categorical variable
    all_levels = [np.unique(x) for x in data_x[:, cat_index].T]
    all_levels_dict = dict(zip(data_df.columns[cat_index], all_levels))

    rmse_ls = []
    for i in range(num_samples):
        file_name = '../samples/{}/{}/sample_{}.csv'.format(save_name, miss_mechanism, i)
        data_x_i = np.loadtxt('../samples/{}/complete/sample_{}.csv'.format(save_name, i), delimiter=",").astype(np.float32)

        miss_data_x = np.loadtxt(file_name, delimiter=",").astype(np.float32)
        data_m = 1 - np.isnan(miss_data_x).astype(np.float32)
        if model_name == "mida":
            imputed_list, loss_list = autoencoder_imputation(miss_data_x, data_m,
                                                               cat_index, num_index,
                                                               all_levels, DAE_parameters, 10)
        if model_name == "gain":
            imputed_list, Gloss_list, Dloss_list = gain(miss_data_x, data_m,
                                                          cat_index, num_index,
                                                          all_levels, gain_parameters, 10)

        for l in range(num_imputations):
            np.savetxt(os.path.join(save_path, "{}/imputed_{}_{}.csv".format(model_name, i, l)), imputed_list[l], delimiter=",")
        print("{} done!".format(i))