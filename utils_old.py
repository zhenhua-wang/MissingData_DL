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
from mlxtend.evaluate import bias_variance_decomp


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


def onehot(data, data_m, all_levels):
    no, dim = data.shape
    enc = OneHotEncoder(categories=all_levels, sparse=False, dtype=np.float32)
    data_x_enc = enc.fit_transform(data)
    n_classes = list(map(lambda x: len(x), all_levels))

    # transform missing indicator
    data_m_enc = np.ones((no, sum(n_classes)), dtype=np.float32)
    col_index = 0
    for j in range(dim):
        data_m_enc[:, col_index:col_index + n_classes[j]] = np.repeat(data_m[:, j].reshape([-1, 1]), n_classes[j],
                                                                      axis=1)
        col_index = col_index + n_classes[j]
    return data_x_enc, data_m_enc, n_classes, enc


def reverse_onehot(data_enc, enc):
    return enc.inverse_transform(data_enc)


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

def barplot_categorical(imputed, sampled, population, varname, modelname):
    no_imputed, _ = imputed.shape
    no_sampled, _ = sampled.shape
    no_pop, _ = population.shape
    imputed_table = imputed.groupby(varname).size() / no_imputed
    sampled_table = sampled.groupby(varname).size() / no_sampled
    pop_table = population.groupby(varname).size() / no_pop

    merged = pd.concat([pd.DataFrame({"type": "imputed", "percentage": imputed_table.values, "levels": imputed_table.index}),
                        pd.DataFrame({"type": "sampled", "percentage": sampled_table.values, "levels": sampled_table.index}),
                        pd.DataFrame({"type": "populat", "percentage": pop_table.values, "levels": pop_table.index})])
    p = sns.catplot(x="levels", y="percentage", hue="type", kind="bar", palette="muted", data=merged)
    p.fig.suptitle(modelname, fontsize=16)
    return p

def barplot_metric(methods_acc, metric_name):
    df_list = []
    for method_acc, method_name in methods_acc:
        df_list.append(pd.DataFrame({"variable": list(method_acc.keys()),
                                     metric_name: list(method_acc.values()),
                                     "method": method_name}))
    p = sns.catplot(x="variable", y=metric_name, hue="method", kind="bar", palette="muted", data=pd.concat(df_list),
                legend_out=False)
    p.fig.set_size_inches(20, 8)
    plt.xticks(rotation=90)
    plt.show()

def onehot_encoding_backup(data, all_levels):
    no, dim = data.shape

    data_m = 1 - np.isnan(data)
    data_filled = np.nan_to_num(data.copy(), 0)
    data_enc = np.empty(shape=(no, np.sum([len(x) for x in all_levels])))
    col_idx = 0
    for j in range(dim):
        colj_nlevel = len(all_levels[j])
        colj = data_filled[:, j].astype(np.int)
        miss_j = np.repeat(data_m[:, j].reshape([-1, 1]), colj_nlevel,axis=1)
        enc_j = np.eye(colj_nlevel)[colj]
        enc_j[miss_j == 0] = np.nan
        data_enc[:, col_idx:(col_idx+colj_nlevel)] = enc_j
        col_idx += colj_nlevel
    return data_enc.astype(np.float32)

def onehot_decoding_backup(data_enc, all_levels):
    col_idx = 0
    no = data_enc.shape[0]
    dim = len(all_levels)

    miss_enc = 1-np.isnan(data_enc)
    data = np.empty(shape=(no, dim), dtype=np.float32)
    for j in range(dim):
        colj_level = len(all_levels[j])
        data_enc_j = data_enc[:, col_idx:(col_idx + colj_level)]
        data_j = np.argmax(data_enc_j, axis=1).astype(np.float32)
        data_m_j = miss_enc[:, col_idx]
        data_j[data_m_j == 0] = np.nan
        data[:, j] = data_j
        col_idx += colj_level
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

# test
#al = np.array([[0, 1, 2, 3], [0, 1]])
#a = np.array([[2, np.nan], [1, 1], [3, 1]])

#print(onehot_encoding(a, al))
#print(onehot_decoding(onehot_encoding(a, al), al))

# sum dict values for two dicts with the same keys
def sum_over_dicts(pre, new):
    if pre:
        sum_dict = dict(zip(new.keys(),
                            np.array(list(pre.values())) +
                            np.array(list(new.values()))))
    else:
        sum_dict = new
    return sum_dict

def get_bins_from_numerical(num_df, bins):
    data_bin_ls = []
    for i in range(len(bins)):
        col = num_df.columns[i]
        data_bin_ls.append(pd.cut(num_df[col], bins=bins[i], labels = False))
    data_bin_df = pd.concat(data_bin_ls, axis=1)
    return data_bin_df

#def get_varname_index(all_levels):

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

house_bins = [np.array([-1., 2., 3., 4., 8.]),
            np.array([0., 2., 3., 4., 20.]),
            np.array([0., 4., 5., 6., 7., 9., 19.]),
            np.array([14., 32., 41., 49., 55., 61., 67., 75., 96.]),
            np.array([-1., 0, 8., 15., 25., 40., 188.]),
            np.array(
                [0.000e+00, 3.400e+01, 5.100e+01, 6.300e+01, 7.600e+01, 9.400e+01, 1.170e+02, 1.640e+02, 2.097e+03]),
            np.array([-1., 0., 19000., 35600., 54000., 85000., 718000.]),
            np.array([-1., 0., 38., 40., 50., 99.])]