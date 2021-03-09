from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tqdm import tqdm

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib


from utils.utils import rmse_loss, get_bins_from_numerical
from evaluation.performance_metric import marginal_estimands, bivariate_estimands, house_bins

# Load data
model_names = ["cart", "rf", "gain_softmax", "dae_softmax"]
num_imputations = 10

# data_name = "house_recoded"
# save_name = "house2"
data_name = "letter"
save_name = "letter"
miss_mechanism = "MCAR"
file_name = 'data/{}.csv'.format(data_name)
data_df = pd.read_csv(file_name)
data_x = data_df.values.astype(np.float32)

if save_name == "house2":
    num_index = list(range(-8, 0))
    cat_index = list(range(-data_df.shape[1], -8))

if save_name == "letter":
    num_index = []
    cat_index = list(range(0, data_x.shape[1]))
if save_name == "spam" or save_name == "breast":
    num_index = list(range(0, data_x.shape[1]))#list(range(40, 48))
    cat_index = []#list(range(0, data_x.shape[1]))#list(range(0, 40))
if save_name == "credit":
    num_index = [0, 4] + list(range(11, data_x.shape[1]))
    cat_index = [1, 2, 3, 5, 6, 7, 8, 9, 10]
if save_name == "news":
    num_index = list(range(0, 11)) + list(range(17, 29)) + list(range(37, data_x.shape[1]))
    cat_index = list(range(11, 17)) + list(range(29, 37))
# Parameters
no, dim = data_x.shape

n_sample = 10
#sample_idx = 0
n_imp = 10

# cat or binned cat
vartype = "cat"
if vartype == "cat":
    index = cat_index
    all_levels = [np.unique(x) for x in data_x[:, cat_index].T]
    all_levels_dict = dict(zip(data_df.columns[cat_index], all_levels))
else:
    index = num_index
    data_num_pop_df = data_df.iloc[:, num_index]
    if save_name == "house2":
        bins = house_bins
    else:
        _, bins = zip(*data_num_pop_df.apply(pd.qcut, 0, q=8, labels=False, retbins=True, duplicates="drop"))
        bins = [np.insert(l, 0, l[0] - 1, axis=0) for l in bins]
    data_bin_pop_df = get_bins_from_numerical(data_num_pop_df, bins)
    all_levels = [np.unique(x) for x in data_bin_pop_df.values.T]
    all_levels_dict = dict(zip(data_df.columns[num_index], all_levels))

# initialize
true_list = {}
cart_list = {}
gain_list = {}
mida_list = {}
for varname in data_df.columns[index]:
    true_list[varname] = np.zeros(shape=len(all_levels_dict[varname]))
    cart_list[varname] = np.zeros(shape=(n_sample, len(all_levels_dict[varname])))
    gain_list[varname] = np.zeros(shape=(n_sample, len(all_levels_dict[varname])))
    mida_list[varname] = np.zeros(shape=(n_sample, len(all_levels_dict[varname])))

for sample_idx in tqdm(range(n_sample)):
    if save_name == "house2":
        data_sample = pd.read_csv("F:/MIDS/FanLi/MissingData_DL/samples/house2_100k/complete/sample_{}.csv".format(sample_idx))
    else:
        data_sample = pd.read_csv(
            "F:/MIDS/FanLi/MissingData_DL/samples/UCI/{}-{}.csv".format(save_name, sample_idx))
    for l in range(n_imp):
        if save_name == "house2":
            data_cart = pd.read_csv('F:/MIDS/FanLi/MissingData_DL/results/house2_100k/MCAR/cart/imputed_{}_{}.csv'.format(sample_idx, l))
            data_gain = pd.read_csv('F:/MIDS/FanLi/MissingData_DL/results/house2_100k/MCAR/gain_softmax/imputed_{}_{}.csv'.format(sample_idx, l), header=None)
            data_mida = pd.read_csv('F:/MIDS/FanLi/MissingData_DL/results/house2_100k/MCAR/dae_softmax/imputed_{}_{}.csv'.format(sample_idx, l), header=None)
        else:
            data_cart = pd.read_csv(
                'F:/MIDS/FanLi/MissingData_DL/results/UCI/{}/{}_cart_{}.csv'.format(save_name, sample_idx, l))
            data_gain = pd.read_csv(
                'F:/MIDS/FanLi/MissingData_DL/results/UCI/{}/{}_gain_softmax_{}.csv'.format(save_name, sample_idx,l),header=None)
            data_mida = pd.read_csv(
                'F:/MIDS/FanLi/MissingData_DL/results/UCI/{}/{}_dae_softmax_{}.csv'.format(save_name, sample_idx, l),header=None)

        data_sample.columns = data_df.columns
        data_cart.columns = data_df.columns
        # colnames(data_rf) = colnames(data_df)
        data_gain.columns = data_df.columns
        data_mida.columns = data_df.columns

        # get cat or binned cat
        if vartype == "cat":
            data_df_cat = data_df[data_df.columns[cat_index]]
            data_cart_cat = data_cart[data_df.columns[cat_index]]
            data_gain_cat = data_gain[data_df.columns[cat_index]]
            data_mida_cat = data_mida[data_df.columns[cat_index]]
        else:
            data_df_num = data_df[data_df.columns[num_index]]
            data_cart_num = data_cart[data_df.columns[num_index]]
            data_gain_num = data_gain[data_df.columns[num_index]]
            data_mida_num = data_mida[data_df.columns[num_index]]

            data_df_cat = get_bins_from_numerical(data_df_num, bins)
            data_cart_cat = get_bins_from_numerical(data_cart_num, bins)
            data_gain_cat = get_bins_from_numerical(data_gain_num, bins)
            data_mida_cat = get_bins_from_numerical(data_mida_num, bins)

        for varname in data_df.columns[index]:
            true_table = data_df_cat[varname].value_counts().sort_index() / data_df_cat.shape[0]
            cart_table = data_cart_cat[varname].value_counts().sort_index().reindex(true_table.index).fillna(0) / data_cart_cat.shape[0]
            gain_table = data_gain_cat[varname].value_counts().sort_index().reindex(true_table.index).fillna(0) / data_gain_cat.shape[0]
            mida_table = data_mida_cat[varname].value_counts().sort_index().reindex(true_table.index).fillna(0) / data_mida_cat.shape[0]

            true_list[varname] = true_table.to_numpy()
            cart_list[varname][sample_idx, :] += cart_table.to_numpy()
            gain_list[varname][sample_idx, :] += gain_table.to_numpy()
            mida_list[varname][sample_idx, :] += mida_table.to_numpy()


for varname in data_df.columns[index]:
    cart_list[varname] = cart_list[varname] / n_imp
    gain_list[varname] = gain_list[varname] / n_imp
    mida_list[varname] = mida_list[varname] / n_imp


diff_df = pd.DataFrame(0, index=data_df.columns[index], columns=["CART", "GAIN", "MIDA"], dtype=np.float64)

for varname in data_df.columns[index]:
    for sample_idx in range(n_sample):
        diff_df.loc[varname, "CART"] += sum(abs(cart_list[varname][sample_idx, :] - true_list[varname]) * true_list[varname])
        diff_df.loc[varname, "GAIN"] += sum(abs(gain_list[varname][sample_idx, :] - true_list[varname]) * true_list[varname])
        diff_df.loc[varname, "MIDA"] += sum(abs(mida_list[varname][sample_idx, :] - true_list[varname]) * true_list[varname])

if save_name == "house2":
    print((diff_df / n_sample).describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).loc[["10%", "25%", "50%", "75%", "90%"]]*100)
else:
    print((diff_df / n_sample).describe().loc[["50%"]]*100)
