import numpy as np
from itertools import combinations
from tqdm import tqdm
from scipy.stats import norm, t

def combinaion_2lists(l1, l2):
    mesh = np.array(np.meshgrid(l1, l2))
    combinations = mesh.T.reshape(-1, 2)
    return combinations

def marginal_estimands(df, all_levels_dict, with_idx = False):
    mar_prob = []
    variable_idx = []
    for col in tqdm(df.columns):
        mar_prob_curr = np.sum(df[col].to_numpy().reshape([-1, 1]) == all_levels_dict[col].reshape(1, -1), axis=0)/df.shape[0]
        mar_prob = mar_prob + mar_prob_curr.tolist()
        variable_idx += [col] * len(all_levels_dict[col])
    mar_prob = np.array(mar_prob)
    mar_var = (mar_prob * (1-mar_prob)) / df.shape[0]
    if with_idx:
        return mar_prob, mar_var, variable_idx
    else:
        return mar_prob, mar_var

def bivariate_estimands(df, all_levels_dict):
    n_row, n_col = df.shape
    bi_combns = list(combinations(df.columns, 2))

    bivar_prob = []
    for col1, col2 in tqdm(bi_combns):
        level_combns = combinaion_2lists(all_levels_dict[col1], all_levels_dict[col2])
        bivar_prob_curr = np.sum((df[col1].to_numpy().reshape([-1, 1]) == level_combns[:, 0].reshape(1, -1)) &
                                 (df[col2].to_numpy().reshape([-1, 1]) == level_combns[:, 1].reshape(1, -1)), axis=0) / n_row
        bivar_prob = bivar_prob + bivar_prob_curr.tolist()

    bivar_prob = np.array(bivar_prob)
    bivar_var = (bivar_prob * (1 - bivar_prob)) / n_row
    return bivar_prob, bivar_var

def complete_CI(prob, var):
    no, num_samples = prob.shape
    CI = np.zeros(shape=(no, num_samples, 2))
    lower = prob + (norm.ppf(0.025) * np.sqrt(var))
    upper = prob - (norm.ppf(0.025) * np.sqrt(var))
    CI[:, :, 0] = lower
    CI[:, :, 1] = upper
    return CI
def imputed_CI(prob, var):
    no, num_samples, num_imputations = prob.shape
    L = num_imputations
    CI = np.zeros(shape=(no, num_samples, 2))
    qbar = np.mean(prob, axis=2)
    b = np.var(prob, ddof=1, axis=2)
    ubar = np.mean(var, axis=2)
    T = ubar + (b*(L+1)/L)
    r = ubar / b
    r[np.isinf(r)] = np.nan
    nu = (L - 1) * ((1 + ((L / (L + 1)) * r)) ** 2)
    lower = qbar + (t.ppf(0.025, nu) * np.sqrt(T))
    upper = qbar - (t.ppf(0.025, nu) * np.sqrt(T))
    CI[:, :, 0] = lower
    CI[:, :, 1] = upper
    return CI, T

def rel_mse(q_bar, q_hat, Q):
    qbar_mse = np.sum((q_bar - np.tile(Q, (q_bar.shape[1], 1)).T)**2, axis=1)
    qhat_mse = np.sum((q_hat - np.tile(Q, (q_hat.shape[1], 1)).T)**2, axis=1)
    return qbar_mse / qhat_mse, qbar_mse, qhat_mse


def rel_mse_bias_var(q_bar, q_hat, Q):
    relmse, qbar_mse, qhat_mse = rel_mse(q_bar, q_hat, Q)
    mse = np.mean((q_bar - np.tile(Q, (q_bar.shape[1], 1)).T)**2, axis=1)
    bias = np.mean(q_bar, axis=1) - Q
    var = np.var(q_bar, axis=1)
    nbias = np.abs(bias / Q)
    nvar = np.abs(var / Q)
    relbias = np.abs(bias / (np.mean(q_hat, axis=1) - Q))
    relvar = np.abs(var / (np.var(q_hat, axis=1)))
    return relmse, nbias, nvar, relbias, relvar

def variance_ratio(T, qhat, Q):
    denom = np.nanmean((qhat - np.tile(Q, (qhat.shape[1], 1)).T) ** 2, axis=1)
    vr = np.nanmean(T, axis=1) / denom
    vr[np.isinf(vr)] = np.nan
    return vr

def interval_length_ratio(qbar_CI, qhat_CI):
    ilr = np.nanmean(qbar_CI[:, :, 1] - qbar_CI[:, :, 0], axis=1) / \
           np.nanmean(qhat_CI[:, :, 1] - qhat_CI[:, :, 0], axis=1)
    ilr[np.isinf(ilr)] = np.nan
    return ilr

def coverage_rate(Q, impute_CI):
    cr = []
    for i in range(impute_CI.shape[0]):
        cr_i = 0
        for j in range(impute_CI.shape[1]):
            lower, upper = impute_CI[i, j]
            if (Q[i] >= lower) & (Q[i] <= upper): cr_i += 1
        cr_i = cr_i / impute_CI.shape[1]
        cr.append(cr_i)
    return cr

house_bins = [np.array([-1., 2., 3., 4., 8.]),
            np.array([0., 2., 3., 4., 20.]),
            np.array([0., 4., 5., 6., 7., 9., 19.]),
            np.array([14., 32., 41., 49., 55., 61., 67., 75., 96.]),
            np.array([-1., 0, 8., 15., 25., 40., 188.]),
            np.array(
                [0.000e+00, 3.400e+01, 5.100e+01, 6.300e+01, 7.600e+01, 9.400e+01, 1.170e+02, 1.640e+02, 2.097e+03]),
            np.array([-1., 0., 19000., 35600., 54000., 85000., 718000.]),
            np.array([-1., 0., 38., 40., 50., 99.])]