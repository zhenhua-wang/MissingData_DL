import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({'font.size': 13})
save_name = "house"
file_name = 'data/house_recoded.csv'
data_df = pd.read_csv(file_name)
data_x = data_df.values.astype(np.float32)
miss_mechanism = "MCAR"
save_path = "./metrics/{}/{}".format(save_name, miss_mechanism)

# load cat
mar_coverage = pd.DataFrame(np.load(os.path.join(save_path, "mar_coverage.npy"), allow_pickle=True).item())
biv_coverage = pd.DataFrame(np.load(os.path.join(save_path, "biv_coverage.npy"), allow_pickle=True).item())
mar_ilr = pd.DataFrame(np.load(os.path.join(save_path, "mar_ilr.npy"), allow_pickle=True).item())
biv_ilr = pd.DataFrame(np.load(os.path.join(save_path, "biv_ilr.npy"), allow_pickle=True).item())
# load cont
mar_bin_coverage = pd.DataFrame(np.load(os.path.join(save_path, "mar_bin_coverage.npy"), allow_pickle=True).item())
biv_bin_coverage = pd.DataFrame(np.load(os.path.join(save_path, "biv_bin_coverage.npy"), allow_pickle=True).item())
mar_bin_ilr = pd.DataFrame(np.load(os.path.join(save_path, "mar_bin_ilr.npy"), allow_pickle=True).item())
biv_bin_ilr = pd.DataFrame(np.load(os.path.join(save_path, "biv_bin_ilr.npy"), allow_pickle=True).item())

# plots
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
graph = sns.boxplot(x="variable", y="value", data=pd.melt(mar_coverage), color="skyblue")
graph.axhline(0.95, ls='--', c="red")
graph.set(xlabel='', ylabel='')
plt.title("Marginal for Categorical variables")
plt.subplot(2, 2, 2)
graph = sns.boxplot(x="variable", y="value", data=pd.melt(biv_coverage), color="skyblue")
graph.axhline(0.95, ls='--', c="red")
graph.set(xlabel='', ylabel='')
plt.title("Bivariate for Categorical variables")
plt.subplot(2, 2, 3)
graph = sns.boxplot(x="variable", y="value", data=pd.melt(mar_bin_coverage), color="skyblue")
graph.axhline(0.95, ls='--', c="red")
graph.set(xlabel='', ylabel='')
plt.title("Marginal for continuous variables")
plt.subplot(2, 2, 4)
graph = sns.boxplot(x="variable", y="value", data=pd.melt(biv_bin_coverage), color="skyblue")
graph.axhline(0.95, ls='--', c="red")
graph.set(xlabel='', ylabel='')
plt.title("Bivariate for continuous variables")
plt.tight_layout()
plt.savefig(os.path.join(save_path, "coverage.png"))
plt.clf()

# ilr
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
graph = sns.boxplot(x="variable", y="value", data=pd.melt(mar_ilr), color="skyblue")
graph.set(xlabel='', ylabel='')
plt.title("Marginal for Categorical variables")
plt.subplot(2, 2, 2)
graph = sns.boxplot(x="variable", y="value", data=pd.melt(biv_ilr), color="skyblue")
graph.set(xlabel='', ylabel='')
plt.title("Bivariate for Categorical variables")
plt.subplot(2, 2, 3)
graph = sns.boxplot(x="variable", y="value", data=pd.melt(mar_bin_ilr), color="skyblue")
graph.set(xlabel='', ylabel='')
plt.title("Marginal for continuous variables")
plt.subplot(2, 2, 4)
graph = sns.boxplot(x="variable", y="value", data=pd.melt(biv_bin_ilr), color="skyblue")
graph.set(xlabel='', ylabel='')
plt.title("Bivariate for continuous variables")
plt.tight_layout()
plt.savefig(os.path.join(save_path, "ilr.png"))
plt.clf()

