import os
import numpy as np
import pandas as pd
import pathlib
from evaluation.performance_metric import complete_CI, imputed_CI, coverage_rate, rel_mse_bias_var, interval_length_ratio, variance_ratio, rel_mse

# Load data
model_names = ["cart", "rf", "gain", "mida"]
miss_mechanism = "MCAR"

save_name = "house"
file_name = '../data/house_recoded.csv'
data_df = pd.read_csv(file_name)
data_x = data_df.values.astype(np.float32)
# Parameters
no, dim = data_x.shape
n = 10000

num_index = list(range(-8, 0))
cat_index = list(range(-data_df.shape[1], -8))

save_path = "../metrics/{}/{}".format(save_name, miss_mechanism)
pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)


# load estimands
if cat_index:
    # population estimands
    mar_Q = np.load(os.path.join(save_path, "mar_Q.npy"))
    mar_Q_var = np.load(os.path.join(save_path, "mar_Q_var.npy"))
    biv_Q = np.load(os.path.join(save_path, "biv_Q.npy"))
    biv_Q_var = np.load(os.path.join(save_path, "biv_Q_var.npy"))
    # qualified index
    mar_index = (mar_Q * n > 10) & ((1 - mar_Q) * n > 10)
    biv_index = (biv_Q * n > 10) & ((1 - biv_Q) * n > 10)
    # premiss estimands
    mar_qhat = np.load(os.path.join(save_path, "mar_qhat.npy"))
    mar_qhat_var = np.load(os.path.join(save_path, "mar_qhat_var.npy"))
    biv_qhat = np.load(os.path.join(save_path, "biv_qhat.npy"))
    biv_qhat_var = np.load(os.path.join(save_path, "biv_qhat_var.npy"))
    # imputed estimands
    mar_prob_impute = np.load(os.path.join(save_path, "mar_prob_impute.npy"), allow_pickle=True).item()
    mar_var_impute = np.load(os.path.join(save_path, "mar_var_impute.npy"), allow_pickle=True).item()
    biv_prob_impute = np.load(os.path.join(save_path, "biv_prob_impute.npy"), allow_pickle=True).item()
    biv_var_impute = np.load(os.path.join(save_path, "biv_var_impute.npy"), allow_pickle=True).item()
    # initialize metrics
    mar_coverage = {}
    biv_coverage = {}
    mar_rel_mse, mar_nbias, mar_nvar, mar_relbias, mar_relvar, mar_ilr, mar_var_ratio, mar_qbar_mse, mar_qhat_mse = \
        {}, {}, {}, {}, {}, {}, {}, {}, {}
    biv_rel_mse, biv_nbias, biv_nvar, biv_relbias, biv_relvar, biv_ilr, biv_var_ratio = \
        {}, {}, {}, {}, {}, {}, {}
if num_index:
    # population estimands
    mar_bin_Q = np.load(os.path.join(save_path, "mar_bin_Q.npy"))
    mar_bin_Q_var = np.load(os.path.join(save_path, "mar_bin_Q_var.npy"))
    biv_bin_Q = np.load(os.path.join(save_path, "biv_bin_Q.npy"))
    biv_bin_Q_var = np.load(os.path.join(save_path, "biv_bin_Q_var.npy"))
    # qualified index
    mar_bin_index = (mar_bin_Q * n > 10) & ((1 - mar_bin_Q) * n > 10)
    biv_bin_index = (biv_bin_Q * n > 10) & ((1 - biv_bin_Q) * n > 10)
    # performance metrics
    mar_bin_qhat = np.load(os.path.join(save_path, "mar_bin_qhat.npy"))
    mar_bin_qhat_var = np.load(os.path.join(save_path, "mar_bin_qhat_var.npy"))
    biv_bin_qhat = np.load(os.path.join(save_path, "biv_bin_qhat.npy"))
    biv_bin_qhat_var = np.load(os.path.join(save_path, "biv_bin_qhat_var.npy"))
    # initial imputed metrics
    mar_bin_prob_impute = np.load(os.path.join(save_path, "mar_bin_prob_impute.npy"), allow_pickle=True).item()
    mar_bin_var_impute = np.load(os.path.join(save_path, "mar_bin_var_impute.npy"), allow_pickle=True).item()
    biv_bin_prob_impute = np.load(os.path.join(save_path, "biv_bin_prob_impute.npy"), allow_pickle=True).item()
    biv_bin_var_impute = np.load(os.path.join(save_path, "biv_bin_var_impute.npy"), allow_pickle=True).item()
    # initialize metrics
    mar_bin_coverage = {}
    biv_bin_coverage = {}
    mar_bin_rel_mse, mar_bin_nbias, mar_bin_nvar, mar_bin_relbias, mar_bin_relvar, mar_bin_ilr, mar_bin_var_ratio =\
        {}, {}, {}, {}, {}, {}, {}
    biv_bin_rel_mse, biv_bin_nbias, biv_bin_nvar, biv_bin_relbias, biv_bin_relvar, biv_bin_ilr, biv_bin_var_ratio = \
        {}, {}, {}, {}, {}, {}, {}


# point estimates for qbar
for model_name in model_names:
    if cat_index:
        mar_qbar = np.mean(mar_prob_impute[model_name], axis=2)
        biv_qbar = np.mean(biv_prob_impute[model_name], axis=2)
        # estimated CIs
        mar_qhat_CI = complete_CI(mar_qhat, mar_qhat_var)
        biv_qhat_CI = complete_CI(biv_qhat, biv_qhat_var)
        mar_impute_CI, mar_T = imputed_CI(mar_prob_impute[model_name], mar_var_impute[model_name])
        biv_impute_CI, biv_T = imputed_CI(biv_prob_impute[model_name], biv_var_impute[model_name])
        # coverages
        mar_coverage[model_name] = coverage_rate(mar_Q[mar_index], mar_impute_CI[mar_index])
        biv_coverage[model_name] = coverage_rate(biv_Q[biv_index], biv_impute_CI[biv_index])
        # rel mses
        _, mar_qbar_mse[model_name], mar_qhat_mse[model_name] = rel_mse(mar_qbar[mar_index], mar_qhat[mar_index], mar_Q[mar_index])
        mar_rel_mse[model_name], mar_nbias[model_name], mar_nvar[model_name], mar_relbias[model_name], mar_relvar[model_name] = \
            rel_mse_bias_var(mar_qbar[mar_index], mar_qhat[mar_index], mar_Q[mar_index])
        mar_ilr[model_name] = interval_length_ratio(mar_impute_CI[mar_index], mar_qhat_CI[mar_index])
        mar_var_ratio[model_name] = variance_ratio(mar_T[mar_index], mar_qhat[mar_index], mar_Q[mar_index])
        biv_rel_mse[model_name], biv_nbias[model_name], biv_nvar[model_name], biv_relbias[model_name], biv_relvar[model_name] = \
            rel_mse_bias_var(biv_qbar[biv_index], biv_qhat[biv_index], biv_Q[biv_index])
        biv_ilr[model_name] = interval_length_ratio(biv_impute_CI[biv_index], biv_qhat_CI[biv_index])
        biv_var_ratio[model_name] = variance_ratio(biv_T[biv_index], biv_qhat[biv_index], biv_Q[biv_index])
    if num_index:
        mar_bin_qbar = np.mean(mar_bin_prob_impute[model_name], axis=2)
        biv_bin_qbar = np.mean(biv_bin_prob_impute[model_name], axis=2)
        # estimated CIs
        mar_bin_qhat_CI = complete_CI(mar_bin_qhat, mar_bin_qhat_var)
        biv_bin_qhat_CI = complete_CI(biv_bin_qhat, biv_bin_qhat_var)
        mar_bin_impute_CI, mar_bin_T = imputed_CI(mar_bin_prob_impute[model_name], mar_bin_var_impute[model_name])
        biv_bin_impute_CI, biv_bin_T = imputed_CI(biv_bin_prob_impute[model_name], biv_bin_var_impute[model_name])
        # coverages
        mar_bin_coverage[model_name] = coverage_rate(mar_bin_Q[mar_bin_index], mar_bin_impute_CI[mar_bin_index])
        biv_bin_coverage[model_name] = coverage_rate(biv_bin_Q[biv_bin_index], biv_bin_impute_CI[biv_bin_index])
        # rel mses
        mar_bin_rel_mse[model_name], mar_bin_nbias[model_name], mar_bin_nvar[model_name], mar_bin_relbias[model_name], mar_bin_relvar[model_name] = \
            rel_mse_bias_var(mar_bin_qbar[mar_bin_index], mar_bin_qhat[mar_bin_index], mar_bin_Q[mar_bin_index])
        mar_bin_ilr[model_name] = interval_length_ratio(mar_bin_impute_CI[mar_bin_index], mar_bin_qhat_CI[mar_bin_index])
        mar_bin_var_ratio[model_name] = variance_ratio(mar_bin_T[mar_bin_index], mar_bin_qhat[mar_bin_index], mar_bin_Q[mar_bin_index])
        biv_bin_rel_mse[model_name], biv_bin_nbias[model_name], biv_bin_nvar[model_name], biv_bin_relbias[model_name], biv_bin_relvar[model_name] = \
            rel_mse_bias_var(biv_bin_qbar[biv_bin_index], biv_bin_qhat[biv_bin_index], biv_bin_Q[biv_bin_index])
        biv_bin_ilr[model_name] = interval_length_ratio(biv_bin_impute_CI[biv_bin_index], biv_bin_qhat_CI[biv_bin_index])
        biv_bin_var_ratio[model_name] = variance_ratio(biv_bin_T[biv_bin_index], biv_bin_qhat[biv_bin_index], biv_bin_Q[biv_bin_index])

if cat_index:
    np.save(os.path.join(save_path, "mar_rel_mse.npy"), mar_rel_mse)
    np.save(os.path.join(save_path, "biv_rel_mse.npy"), biv_rel_mse)
    np.save(os.path.join(save_path, "mar_coverage.npy"), mar_coverage)
    np.save(os.path.join(save_path, "biv_coverage.npy"), biv_coverage)
    np.save(os.path.join(save_path, "mar_nbias.npy"), mar_nbias)
    np.save(os.path.join(save_path, "mar_nvar.npy"), mar_nvar)
    np.save(os.path.join(save_path, "mar_relbias.npy"), mar_relbias)
    np.save(os.path.join(save_path, "mar_relvar.npy"), mar_relvar)
    np.save(os.path.join(save_path, "mar_ilr.npy"), mar_ilr)
    np.save(os.path.join(save_path, "mar_var_ratio.npy"), mar_var_ratio)
    np.save(os.path.join(save_path, "biv_nbias.npy"), biv_nbias)
    np.save(os.path.join(save_path, "biv_nvar.npy"), biv_nvar)
    np.save(os.path.join(save_path, "biv_relbias.npy"), biv_relbias)
    np.save(os.path.join(save_path, "biv_relvar.npy"), biv_relvar)
    np.save(os.path.join(save_path, "biv_ilr.npy"), biv_ilr)
    np.save(os.path.join(save_path, "biv_var_ratio.npy"), biv_var_ratio)
if num_index:
    np.save(os.path.join(save_path, "mar_bin_rel_mse.npy"), mar_bin_rel_mse)
    np.save(os.path.join(save_path, "biv_bin_rel_mse.npy"), biv_bin_rel_mse)
    np.save(os.path.join(save_path, "mar_bin_coverage.npy"), mar_bin_coverage)
    np.save(os.path.join(save_path, "biv_bin_coverage.npy"), biv_bin_coverage)
    np.save(os.path.join(save_path, "mar_bin_nbias.npy"), mar_bin_nbias)
    np.save(os.path.join(save_path, "mar_bin_nvar.npy"), mar_bin_nvar)
    np.save(os.path.join(save_path, "mar_bin_relbias.npy"), mar_bin_relbias)
    np.save(os.path.join(save_path, "mar_bin_relvar.npy"), mar_bin_relvar)
    np.save(os.path.join(save_path, "mar_bin_ilr.npy"), mar_bin_ilr)
    np.save(os.path.join(save_path, "mar_bin_var_ratio.npy"), mar_bin_var_ratio)
    np.save(os.path.join(save_path, "biv_bin_nbias.npy"), biv_bin_nbias)
    np.save(os.path.join(save_path, "biv_bin_nvar.npy"), biv_bin_nvar)
    np.save(os.path.join(save_path, "biv_bin_relbias.npy"), biv_bin_relbias)
    np.save(os.path.join(save_path, "biv_bin_relvar.npy"), biv_bin_relvar)
    np.save(os.path.join(save_path, "biv_bin_ilr.npy"), biv_bin_ilr)
    np.save(os.path.join(save_path, "biv_bin_var_ratio.npy"), biv_bin_var_ratio)
