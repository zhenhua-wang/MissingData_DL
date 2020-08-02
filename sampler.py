from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd

from utils import sample_batch_index, binary_sampler
from tqdm import trange

if __name__ == '__main__':
    # Load data
    file_name = 'data/house.csv'
    house_df = pd.read_csv(file_name)
    no, dim = house_df.shape

    data_x = house_df.values.astype(np.float32)
    num_samples = 200

    miss_rate = 0.3
    for i in trange(num_samples):
        # random samples
        sample_idx = sample_batch_index(no, 10000)
        data_x_i = data_x[sample_idx, :]
        no_i, dim_i = data_x_i.shape
        np.savetxt("./samples/complete/sample_{}.csv".format(i), data_x_i, delimiter=",")

        # Introduce missing data
        data_m = binary_sampler(1 - miss_rate, no_i, dim_i)
        miss_data_x = data_x_i.copy()
        miss_data_x[data_m == 0] = np.nan
        np.savetxt("./samples/MCAR/sample_{}.csv".format(i), miss_data_x, delimiter=",")

