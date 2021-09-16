# MissingData_DL

Author: Zhenhua Wang, Olanrewaju Akande, Jason Poulos and Fan Li

## Paper: 
Are deep learning models superior for missing data imputation in large surveys? Evidence from an empirical comparison: 

## Datasets:
  - household: https://www2.census.gov/programs-surveys/acs/data/pums/2019/1-Year/
    - processed data(https://drive.google.com/file/d/1j9KHLJeI4JPyjWcQ5r78hxG5493Msg8l/view?usp=sharing)
  - spam: https://archive.ics.uci.edu/ml/datasets/Spambase
  - letter: https://archive.ics.uci.edu/ml/datasets/Letter+Recognition
  - breast: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
  - credit: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
  - news: https://archive.ics.uci.edu/ml/datasets/online+news+popularity
## Usage
  1. Use sampler.py to create samples with MCAR.
  2. Use main.py to impute the missing dataset. 
  3. To evaluate the performance of missing imputation, we first need to calculate the estimands in the poputaion dataset, the complete sample dataset and the imputed data using evaluation/calculate_estimands.py. Next, we use evaluation/evaluate_estimands.py to calcaute the performance metrics.
  4. To display the performance metrics, we use plot_figures.py and plot_tables.py.
