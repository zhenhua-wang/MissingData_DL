'''Utility functions for GAIN.

(1) normalization: MinMax Normalizer
(2) renormalization: Recover the data from normalzied data
(3) rounding: Handlecategorical variables after imputation
(4) rmse_loss: Evaluate imputed data in terms of RMSE
(5) xavier_init: Xavier initialization
(6) binary_sampler: sample binary random variables
(7) uniform_sampler: sample uniform random variables
(8) sample_batch_index: sample random batch index
'''

# Necessary packages
import numpy as np
import tensorflow.compat.v1 as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder


def normalization(data):
    '''Normalize data in [0, 1] range.
  
  Args:
    - data: original data
  
  Returns:
    - norm_data: normalized data
    - norm_parameters: min_val, max_val for each feature for renormalization
  '''

    # Parameters
    _, dim = data.shape
    norm_data = data.copy()

    # MixMax normalization
    min_val = np.zeros(dim)
    max_val = np.zeros(dim)

    # For each dimension
    for i in range(dim):
        min_val[i] = np.nanmin(norm_data[:, i])
        norm_data[:, i] = norm_data[:, i] - np.nanmin(norm_data[:, i])
        max_val[i] = np.nanmax(norm_data[:, i])
        norm_data[:, i] = norm_data[:, i] / (np.nanmax(norm_data[:, i]) + 1e-6)

        # Return norm_parameters for renormalization
    norm_parameters = {'min_val': min_val,
                       'max_val': max_val}

    return norm_data, norm_parameters


def renormalization(norm_data, norm_parameters):
    '''Renormalize data from [0, 1] range to the original range.
  
  Args:
    - norm_data: normalized data
    - norm_parameters: min_val, max_val for each feature for renormalization
  
  Returns:
    - renorm_data: renormalized original data
  '''

    min_val = norm_parameters['min_val']
    max_val = norm_parameters['max_val']

    _, dim = norm_data.shape
    renorm_data = norm_data.copy()

    for i in range(dim):
        renorm_data[:, i] = renorm_data[:, i] * (max_val[i] + 1e-6)
        renorm_data[:, i] = renorm_data[:, i] + min_val[i]

    return renorm_data


def rounding(imputed_data, data_x):
    '''Round imputed data for categorical variables.
  
  Args:
    - imputed_data: imputed data
    - data_x: original data with missing values
    
  Returns:
    - rounded_data: rounded imputed data
  '''

    _, dim = data_x.shape
    rounded_data = imputed_data.copy()

    for i in range(dim):
        temp = data_x[~np.isnan(data_x[:, i]), i]
        # Only for the categorical variable
        if len(np.unique(temp)) < 10:
            rounded_data[:, i] = np.round(rounded_data[:, i])

    return rounded_data


def rmse_loss(ori_data, imputed_data, data_m):
    '''Compute RMSE loss between ori_data and imputed_data
  
  Args:
    - ori_data: original data without missing values
    - imputed_data: imputed data
    - data_m: indicator matrix for missingness
    
  Returns:
    - rmse: Root Mean Squared Error
  '''

    ori_data, _ = normalization(ori_data)
    imputed_data, _ = normalization(imputed_data)

    # Only for missing values
    nominator = np.sum(((1 - data_m) * ori_data - (1 - data_m) * imputed_data) ** 2)
    denominator = np.sum(1 - data_m)

    rmse = np.sqrt(nominator / float(denominator))

    return rmse


def xavier_init(size):
    '''Xavier initialization.
  
  Args:
    - size: vector size
    
  Returns:
    - initialized random vector.
  '''
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


def binary_sampler(p, rows, cols):
    '''Sample binary random variables.
  
  Args:
    - p: probability of 1
    - rows: the number of rows
    - cols: the number of columns
    
  Returns:
    - binary_random_matrix: generated binary random matrix.
  '''
    unif_random_matrix = np.random.uniform(0., 1., size=[rows, cols])
    binary_random_matrix = 1 * (unif_random_matrix < p)
    return binary_random_matrix.astype('float32')


def uniform_sampler(low, high, rows, cols):
    '''Sample uniform random variables.
  
  Args:
    - low: low limit
    - high: high limit
    - rows: the number of rows
    - cols: the number of columns
    
  Returns:
    - uniform_random_matrix: generated uniform random matrix.
  '''
    return np.random.uniform(low, high, size=[rows, cols]).astype('float32')


def uniform_categorical_sampler(n_classes, rows):
    return np.array([np.random.choice(n, rows) for n in n_classes]).T

def sample_batch_index(total, batch_size):
    '''Sample index of the mini-batch.
  
  Args:
    - total: total number of samples
    - batch_size: batch size
    
  Returns:
    - batch_idx: batch index
  '''
    total_idx = np.random.permutation(total)
    batch_idx = total_idx[:batch_size]
    return batch_idx

def initial_imputation(data_raw, cat_index, num_index):
    data = data_raw.copy()
    # replace nan in categorical variable by the most frequent value
    if cat_index:
        common_value = np.apply_along_axis(lambda x: np.bincount(x[~np.isnan(x)].astype(np.int)).argmax(), 0,
                                           data[:, cat_index])
        for j in range(len(cat_index)):
            data[np.isnan(data[:, cat_index[j]]), cat_index[j]] = common_value[j]
    # replace nan in numerical variable by its mean
    if num_index:
        mean_value = np.nanmean(data[:, num_index], axis=0)
        for j in range(len(num_index)):
            data[np.isnan(data[:, num_index[j]]), num_index[j]] = mean_value[j]
    return data

def onehot_encoding(data, data_m, all_levels, has_miss=False):
    no, dim = data.shape

    data_filled = np.nan_to_num(data.copy(), 0)
    data_enc = np.empty(shape=(no, np.sum([len(x) for x in all_levels])), dtype=np.float32)
    data_m_enc = np.empty(shape=(no, np.sum([len(x) for x in all_levels])), dtype=np.float32)
    col_idx = 0
    for j in range(dim):
        colj_nlevel = len(all_levels[j])
        colj = data_filled[:, j].astype(np.int)
        miss_j = np.repeat(data_m[:, j].reshape([-1, 1]), colj_nlevel,axis=1)
        enc_j = np.eye(colj_nlevel)[colj]
        if has_miss:
            enc_j[miss_j == 0] = np.nan
        data_enc[:, col_idx:(col_idx+colj_nlevel)] = enc_j
        data_m_enc[:, col_idx:(col_idx + colj_nlevel)] = miss_j
        col_idx += colj_nlevel
    return data_enc, data_m_enc

def onehot_decoding(data_enc, data_m_enc, all_levels, has_miss=False):
    col_idx = 0
    no = data_enc.shape[0]
    dim = len(all_levels)

    miss_enc = data_m_enc
    data = np.empty(shape=(no, dim), dtype=np.float32)
    for j in range(dim):
        colj_level = len(all_levels[j])
        data_enc_j = data_enc[:, col_idx:(col_idx + colj_level)]
        data_j = np.argmax(data_enc_j, axis=1).astype(np.float32)
        data_m_j = miss_enc[:, col_idx]
        if has_miss:
            data_j[data_m_j == 0] = np.nan
        data[:, j] = data_j
        col_idx += colj_level
    return data

def get_bins_from_numerical(num_df, bins):
    data_bin_ls = []
    for i in range(len(bins)):
        col = num_df.columns[i]
        data_bin_ls.append(pd.cut(num_df[col], bins=bins[i], labels = False))
    data_bin_df = pd.concat(data_bin_ls, axis=1)
    return data_bin_df


def table_to_latex(mar_table, bias_table, metric_name, variable_type, float_format = "%.2e", save_mode = "w", save_loc = 'mytable.tex', percentage=False):
    tex_table = pd.concat({'Marginal':mar_table, 'Bivariate':bias_table}, axis=1)
    if percentage:
        tex_table *= 100
    tex_table = tex_table.rename(index={0.10: "10%",
                            0.25: "25%",
                            0.50: "50%",
                            0.75: "75%",
                            0.90: "90%"})
    with open(save_loc, save_mode) as tf:
        tex = tex_table.to_latex(float_format = float_format,
                              multicolumn_format = "c",
                              #label = "",
                              caption = "Distributions of {} for {} variables when $n=10000$ and 30\% values MCAR.".format(metric_name, variable_type))
        # tex = tex.replace('\\midrule', '\\hline\n\\midrule')
        # tex = tex.replace('\\bottomrule', '\\bottomrule\n\\hline')
        # tex = tex.replace('{lrrrrrrrr}', '{lrrrrrrrr}\n\hline\hline')
        tex = tex.replace('{Bivariate} \\\\', '{Bivariate} \\\\\n\cline{2-9}')
        tf.write(tex + "\n")