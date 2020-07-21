from itertools import combinations

def marginal_probability(df):
    n_row, n_col = df.shape

    mar_prob = []
    for col in df.columns:
        mar_prob = mar_prob + (list(df.groupby(col).size() / n_row))

    return mar_prob

def bivariate_probability(df):
    n_row, n_col = df.shape
    bi_combns = list([list(cbn) for cbn in combinations(df.columns, 2)])

    bivar_prob = []
    for cbn in bi_combns:
        bivar_prob = bivar_prob + (list(df.groupby(cbn).size() / n_row))

    return bivar_prob