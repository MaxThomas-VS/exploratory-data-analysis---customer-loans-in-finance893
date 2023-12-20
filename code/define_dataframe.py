# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.0
# ---

# +
import db_utils as dbu
import pandas as pd
import numpy as np 
from scipy import stats
import matplotlib.pyplot as plt


# -

# Load raw data from csv

df = pd.read_csv('../data/loan_payments-raw.csv')
data_transform = dbu.DataTransform()
data_frame_transform = dbu.DataFrameTransform()
df_info = dbu.DataFrameInfo()
plotter = dbu.Plotter()


# Inspecting the dataframe we see over 50000 entries for 44 columns
# Upon inspection, 'Unnamed: 0' duplicates the index, so we drop it.

df.info()
print(df['Unnamed: 0'].head(5))
df.drop(columns='Unnamed: 0', inplace=True)

# From prior knowledge of the dataset, we expect columns to be a mixture of categorical, numerical, and dates.
# We use the data_transform class to correct the dytpes.
# +
categorical_columns = ['grade', 'sub_grade', 'employment_length', 'home_ownership','verification_status','loan_status','payment_plan','purpose','term', 'collections_12_mths_ex_med','inq_last_6mths','delinq_2yrs','inq_last_6mths', 'policy_code','application_type']
data_transform.MakeCategorical(df, categorical_columns)

datenum_columns = ['issue_date', 'earliest_credit_line', 'last_payment_date', 'next_payment_date', 'last_credit_pull_date']
data_transform.Dates2Datetimes(df, datenum_columns)
# -

# We also see that sseveral columns have null values.
# Four columns stand out with obviously high nan counts, and we drop these.
# Three columns have less than 1% NaN, so we drop rows from these
# We are left with columns that have too many missing values to drop either the rows or columns.
# +
df_info.PrintNaNFractions(df)
data_frame_transform.DropColsWithNaN(df, ['mths_since_last_delinq',
                                          'mths_since_last_record',
                                          'next_payment_date',
                                          'mths_since_last_major_derog'])

data_frame_transform.DropRowsWithNaN(df, ['last_payment_date',
                                          'last_credit_pull_date',
                                          'collections_12_mths_ex_med'])

df_info.PrintNaNFractions(df)
# -


# We visualise these columns to impute these NaN thoughtfully.
plotter.InspectNaN(df, ['matrix','heatmap'])

# The NaN look uncorrelated accross the data, and a heatmap of the correlation confirms this.
# Next, we check the basic stats of each variable.
df_info.PrintColumnInfo(df, 'funded_amount')
df_info.PrintColumnInfo(df, 'term')
df_info.PrintColumnInfo(df, 'int_rate')
df_info.PrintColumnInfo(df, 'employment_length')

# We check for correlations between the numerical variables, to see if we have options to impute by regression.
isnumeric = df_info.IsNumeric(df)
plotter.CorrelationHeatmap(df, isnumeric)

# Interest rate is poorly correlated with all variables (max or around 0.5).
# We therefore impute this value with the median, which we see is similar to the mean.
data_frame_transform.ImputeNaN(df, 'int_rate', 'median')

# employment_length has 11 categories, none of which are dominant. 
# We therefore add an 'Unknown' category, rather than potentially biasing by imputing.
data_frame_transform.ImputeNaN(df, 'employment_length', 'Unknown')

# funded_amount is strongly correlated with funded_amount_inv and installment.
# We perform a multiple linear regression of 
# **funded_amount ~ 'funded_amount', 'funded_amount_inv', 'instalment','total_rec_int','total_rec_prncp','total_payment_inv','total_payment'**
# to impute the values.
# +
predictors = ['funded_amount_inv', 'instalment','total_rec_int','total_rec_prncp','total_payment_inv','total_payment']

mlr_mask = data_frame_transform.DefineMLR2Impute(df, 'funded_amount', predictors)

data_frame_transform.ImputeNaNMLR(df, 'funded_amount', predictors, mlr_mask)
# -

# term has two values, 36 months or 60 months. It is an important variable for later analysis.
# They have similar frequencies so imputing with the most common is risky.
# We therefore iteratively imputate term using a random forest method.
data_frame_transform.ImputeTerm(df)

# Now we've dealt with all the NaN, so we reindex to clean up.
df_info.PrintNaNFractions(df)
df.reset_index(inplace=True)

# Next we inspect the numerical data for skew. Where data are skewed away from normality, we test some transformations to reduce the skew.
# skews with magnitudes less than 0.5 are acceptable, between 0.5 and 1 are moderate, and greater than 1 are severely skewed.
#  
# +
skew = df.skew(numeric_only=True) 
very_skewed = list(skew.index[abs(skew)>=1])
moderately_skewed = list(skew.index[(abs(skew)>=0.5) & (abs(skew)<1)])
all_skewed = list(skew.index[abs(skew)>=0.5])

print('Moderately skewed:')
print(moderately_skewed)
print('')
print('Very skewed:')
print(very_skewed)
# -

# We test log, box-cox, and yeo-johnson transformations on each skewed variable.
# The yeo-johnson transofrmation performs best in all cases where a transofrmation is reasonable.
all_skewed.remove('member_id')
all_skewed.remove('id')
for col in all_skewed:
    plotter.TransformTest(df, col=col)


# Several variables look normal under the Yeo Johnson transformation:
# *loan_amount, funded_amount, funded_amount_inv, instalment, annual_inc, open_accounts, total_accounts, total_payment, total_payment_inv, total_rec_prncp, total_rec_int*
# We add columns for each of these with the YJ transformation applied.
# The other variables are not amenable to transformation, and are often bimodal, so we leave these untouched. 
# +
to_transform = ['loan_amount', 'funded_amount', 'funded_amount_inv', 'instalment', 'annual_inc', 'open_accounts', 'total_accounts', 'total_payment', 'total_payment_inv', 'total_rec_prncp', 'total_rec_int']
for col in to_transform:
    new_col = col + '-yc'
    df[new_col] = pd.Series(stats.yeojohnson(df[col])[0]).rename(new_col, inplace=True)

df.info()
# -

# Next we visualise, identify, and remove outliers.
# As a threshold, we remove outliers that are more extreme than 3 standard deviations from the mean.
# We need an approximately normal distribution to apply the zscore criteria, so we use only columns which we performed the YC transformation on.
# +
to_check_outliers = [tt + '-yc' for tt in to_transform]
rows_with_no_outliers = (np.abs(stats.zscore(df[to_check_outliers])) < 3).all(axis=1)
df_test = df[rows_with_no_outliers]
df_test.info()

plotter.CheckOutlierRemoval(df, df_test, to_check_outliers)
# -

# We see that the oulier quality control step removed less than 2% of the data.
# Histograms are similar before and after removal, and we see fewer outlier in the box and whisker plots.
# We therefore go ahead and overwrite df with df_test, which has the outlier removed.
print('%s %% of data removed as outliers.' % (100*(len(df)-len(df_test))/len(df)))
df = df_test

# Next, we check for correlation between columns with approximately normal data.
plotter.CorrelationHeatmap(df, to_check_outliers)

# Several variable pairs have correlations > 0.9, with some rounding off to 1.
# Rather than dropping variables now, we will use this figure as a reference during the analysis step.

# Finally, we save the dataframe to a csv for loading in the analysis step.
df.to_csv('../data/loan_payments-clean.csv')
df.to_pickle('../data/loan_payments-clean.pkl')
