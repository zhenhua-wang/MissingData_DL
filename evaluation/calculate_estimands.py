from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pandas as pd
import pathlib

from utils.utils import rmse_loss, get_bins_from_numerical
from evaluation.performance_metric import marginal_estimands, bivariate_estimands, house_bins

# Load data
model_names = ["cart", "rf", "gain", "mida"]
num_samples = 100
num_imputations = 10

save_name = "house"
miss_mechanism = "MCAR"
file_name = '../data/house_recoded.csv'
data_df = pd.read_csv(file_name)
data_x = data_df.values.astype(np.float32)

num_index = list(range(-8, 0))
cat_index = list(range(-data_df.shape[1], -8))

# Parameters
no, dim = data_x.shape

# seperate categorical variables and numerical variables
if cat_index:
    data_cat_pop_df = data_df.iloc[:, cat_index]
    # get all possible levels for categorical variable
    all_levels = [np.unique(x) for x in data_x[:, cat_index].T]
    all_levels_dict = dict(zip(data_df.columns[cat_index], all_levels))
    # population estimands
    mar_Q, mar_Q_var = marginal_estimands(data_cat_pop_df, all_levels_dict)
    biv_Q, biv_Q_var = bivariate_estimands(data_cat_pop_df, all_levels_dict)
    # qualified index
    mar_index = (mar_Q * no > 10) & ((1 - mar_Q) * no > 10)
    biv_index = (biv_Q * no > 10) & ((1 - biv_Q) * no > 10)
    # performance metrics
    mar_qhat = np.empty(shape=(mar_Q.shape[0], num_samples))
    mar_qhat_var = np.empty(shape=(mar_Q_var.shape[0], num_samples))
    biv_qhat = np.empty(shape=(biv_Q.shape[0], num_samples))
    biv_qhat_var = np.empty(shape=(biv_Q_var.shape[0], num_samples))
    # initial imputed metrics
    mar_prob_impute = {}
    mar_var_impute = {}
    biv_prob_impute = {}
    biv_var_impute = {}


if num_index:
    data_num_pop_df = data_df.iloc[:, num_index]
    if save_name != "house2":
        data_bin_pop_ls, bins = zip(*data_num_pop_df.apply(pd.qcut, 0, q=8, labels = False, retbins=True, duplicates="drop"))
        data_bin_pop_df = pd.concat(data_bin_pop_ls, axis=1)
    else:
        bins = house_bins
        data_bin_pop_df = get_bins_from_numerical(data_num_pop_df, house_bins)
    # get all possible levels
    bin_all_levels = [np.unique(x) for x in data_bin_pop_df.values.T]
    bin_all_levels_dict = dict(zip(data_df.columns[num_index], bin_all_levels))
    # population estimands
    mar_bin_Q, mar_bin_Q_var = marginal_estimands(data_bin_pop_df, bin_all_levels_dict)
    biv_bin_Q, biv_bin_Q_var = bivariate_estimands(data_bin_pop_df, bin_all_levels_dict)
    # qualified index
    mar_bin_index = (mar_bin_Q * no > 10) & ((1 - mar_bin_Q) * no > 10)
    biv_bin_index = (biv_bin_Q * no > 10) & ((1 - biv_bin_Q) * no > 10)
    # performance metrics
    mar_bin_qhat = np.empty(shape=(mar_bin_Q.shape[0], num_samples))
    mar_bin_qhat_var = np.empty(shape=(mar_bin_Q_var.shape[0], num_samples))
    biv_bin_qhat = np.empty(shape=(biv_bin_Q.shape[0], num_samples))
    biv_bin_qhat_var = np.empty(shape=(biv_bin_Q_var.shape[0], num_samples))
    # initial imputed metrics
    mar_bin_prob_impute = {}
    mar_bin_var_impute = {}
    biv_bin_prob_impute = {}
    biv_bin_var_impute = {}

mse = {}

for model_name in model_names:
    if cat_index:
        mar_prob_impute[model_name] = np.empty(shape=(mar_Q.shape[0], num_samples, num_imputations))
        mar_var_impute[model_name] = np.empty(shape=(mar_Q_var.shape[0], num_samples, num_imputations))
        biv_prob_impute[model_name] = np.empty(shape=(biv_Q.shape[0], num_samples, num_imputations))
        biv_var_impute[model_name] = np.empty(shape=(biv_Q_var.shape[0], num_samples, num_imputations))
    if num_index:
        mar_bin_prob_impute[model_name] = np.empty(shape=(mar_bin_Q.shape[0], num_samples, num_imputations))
        mar_bin_var_impute[model_name] = np.empty(shape=(mar_bin_Q_var.shape[0], num_samples, num_imputations))
        biv_bin_prob_impute[model_name] = np.empty(shape=(biv_bin_Q.shape[0], num_samples, num_imputations))
        biv_bin_var_impute[model_name] = np.empty(shape=(biv_bin_Q_var.shape[0], num_samples, num_imputations))

    # acc[model_name] = []
    mse[model_name] = []

for i in range(num_samples):
    # load samples
    data_i = np.loadtxt('../samples/{}/complete/sample_{}.csv'.format(save_name, i),
                        delimiter=",").astype(np.float32)
    data_miss_i = np.loadtxt('../samples/{}/{}/sample_{}.csv'.format(save_name, miss_mechanism, i),
                             delimiter=",").astype(np.float32)
    data_m = 1 - np.isnan(data_miss_i).astype(np.float32)
    # seperate categorical variables and numerical variables
    if cat_index:
        data_cat = data_i[:, cat_index]
        data_m_cat = data_m[:, cat_index]
        data_cat_df = pd.DataFrame(data=data_cat,
                                   index=list(range(data_cat.shape[0])),
                                   columns=data_df.columns[cat_index])
        # marginal prob and bivariate prob before introduce missingness
        mar_qhat[:, i], mar_qhat_var[:, i] = marginal_estimands(data_cat_df, all_levels_dict)
        biv_qhat[:, i], biv_qhat_var[:, i] = bivariate_estimands(data_cat_df, all_levels_dict)
    if num_index:
        data_num = data_i[:, num_index]
        data_m_num = data_m[:, num_index]
        data_num_df = pd.DataFrame(data=data_num,
                                index=list(range(data_num.shape[0])),
                                columns=data_df.columns[num_index])
        data_bin_df = get_bins_from_numerical(data_num_df, bins)
        # marginal prob and bivariate prob before introduce missingness
        mar_bin_qhat[:, i], mar_bin_qhat_var[:, i] = marginal_estimands(data_bin_df, bin_all_levels_dict)
        biv_bin_qhat[:, i], biv_bin_qhat_var[:, i] = bivariate_estimands(data_bin_df, bin_all_levels_dict)

    for model_name in model_names:
        print("{}th sample, model: {}".format(i, model_name))
        for l in range(num_imputations):
            # loading imputations
            if model_name == "gain" or model_name == "mida":
                data_imputed = np.loadtxt('../results/{}/{}/{}/imputed_{}_{}.csv'.format(save_name, miss_mechanism, model_name, i, l),delimiter=",").astype (np.float32)
            if model_name == "cart" or model_name =="rf":
                data_imputed = pd.read_csv('../results/{}/{}/{}/imputed_{}_{}.csv'.format(save_name, miss_mechanism, model_name, i, l)).values.astype(np.float32)
            # report accuracy
            mse[model_name].append(rmse_loss(data_i, data_imputed, data_m))
            # seperate categorical variables an d numerical variables
            if cat_index:
                imputed_cat = data_imputed[:, cat_index]
                imputed_cat_df = pd.DataFrame(data=imputed_cat,
                                              index=list(range(imputed_cat.shape[0])),
                                              columns=data_df.columns[cat_index])
                # get imputed marginal prob and biviate prob for ith sample
                mar_prob_impute[model_name][:, i, l], mar_var_impute[model_name][:, i, l] = marginal_estimands(
                    imputed_cat_df, all_levels_dict)
                biv_prob_impute[model_name][:, i, l], biv_var_impute[model_name][:, i, l] = bivariate_estimands(
                    imputed_cat_df, all_levels_dict)
            if num_index:
                imputed_num = data_imputed[:, num_index]
                imputed_num_df = pd.DataFrame(data=imputed_num,
                                           index=list(range(imputed_num.shape[0])),
                                           columns=data_df.columns[num_index])
                imputed_bin_df = get_bins_from_numerical(imputed_num_df, bins)
                # get imputed marginal prob and biviate prob for ith sample
                mar_bin_prob_impute[model_name][:, i, l], mar_bin_var_impute[model_name][:, i, l] = marginal_estimands(imputed_bin_df, bin_all_levels_dict)
                biv_bin_prob_impute[model_name][:, i, l], biv_bin_var_impute[model_name][:, i, l] = bivariate_estimands(imputed_bin_df, bin_all_levels_dict)
            pass
        pass
    pass

# save estimands
save_path = "../metrics/{}/{}".format(save_name, miss_mechanism)
pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
if cat_index:
    # population estimands
    np.save(os.path.join(save_path, "mar_Q"), mar_Q)
    np.save(os.path.join(save_path, "mar_Q_var"), mar_Q_var)
    np.save(os.path.join(save_path, "biv_Q"), biv_Q)
    np.save(os.path.join(save_path, "biv_Q_var"), biv_Q_var)
    # premiss estimands
    np.save(os.path.join(save_path, "mar_qhat"), mar_qhat)
    np.save(os.path.join(save_path, "mar_qhat_var"), mar_qhat_var)
    np.save(os.path.join(save_path, "biv_qhat"), biv_qhat)
    np.save(os.path.join(save_path, "biv_qhat_var"), biv_qhat_var)
    # imputed estimands
    np.save(os.path.join(save_path, "mar_prob_impute"), mar_prob_impute)
    np.save(os.path.join(save_path, "mar_var_impute"), mar_var_impute)
    np.save(os.path.join(save_path, "biv_prob_impute"), biv_prob_impute)
    np.save(os.path.join(save_path, "biv_var_impute"), biv_var_impute)
if num_index:
    # population estimands
    np.save(os.path.join(save_path, "mar_bin_Q"), mar_bin_Q)
    np.save(os.path.join(save_path, "mar_bin_Q_var"), mar_bin_Q_var)
    np.save(os.path.join(save_path, "biv_bin_Q"), biv_bin_Q)
    np.save(os.path.join(save_path, "biv_bin_Q_var"), biv_bin_Q_var)
    # performance metrics
    np.save(os.path.join(save_path, "mar_bin_qhat"), mar_bin_qhat)
    np.save(os.path.join(save_path, "mar_bin_qhat_var"), mar_bin_qhat_var)
    np.save(os.path.join(save_path, "biv_bin_qhat"), biv_bin_qhat)
    np.save(os.path.join(save_path, "biv_bin_qhat_var"), biv_bin_qhat_var)
    # initial imputed metrics
    np.save(os.path.join(save_path, "mar_bin_prob_impute"), mar_bin_prob_impute)
    np.save(os.path.join(save_path, "mar_bin_var_impute"), mar_bin_var_impute)
    np.save(os.path.join(save_path, "biv_bin_prob_impute"), biv_bin_prob_impute)
    np.save(os.path.join(save_path, "biv_bin_var_impute"), biv_bin_var_impute)

np.save(os.path.join(save_path, "mse"), mse)