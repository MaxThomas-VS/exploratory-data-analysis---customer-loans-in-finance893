#%%
import db_utils as dbu
import pandas as pd


# %% [markdown]
# Load raw data from csv
df = pd.read_csv('../data/loan_payments-raw.csv')
data_transform = dbu.DataTransform()
data_frame_transform = dbu.DataFrameTransform()
df_info = dbu.DataFrameInfo()
plotter = dbu.Plotter()


# %% [markdown]
# Inspecting the dataframe we see over 50000 entries for 44 columns
# Upon inspection, 'Unnamed: 0' duplicates the index, so we drop it.
df.info()
print(df['Unnamed: 0'].head(5))
df.drop(columns='Unnamed: 0', inplace=True)

# %% [markdown]
# From prior knowledge of the dataset, we expect columns to be a mixture of categorical, numerical, and dates.
# We use the data_transform class to correct the dytpes.
categorical_columns = ['grade', 'sub_grade', 'employment_length', 'home_ownership','verification_status','loan_status','payment_plan','purpose','term', 'collections_12_mths_ex_med','inq_last_6mths','delinq_2yrs','inq_last_6mths', 'policy_code','application_type']
data_transform.MakeCategorical(df, categorical_columns)

datenum_columns = ['issue_date', 'earliest_credit_line', 'last_payment_date', 'next_payment_date', 'last_credit_pull_date']
data_transform.Dates2Datetimes(df, datenum_columns)

# %% [markdown]
# We also see that sseveral columns have null values.
# Four columns stand out with obviously high nan counts, and we drop these.
# Three columns have less than 1% NaN, so we drop rows from these
# We are left with columns that have too many missing values to drop either the rows or columns.
df_info.PrintNaNFractions(df)
data_frame_transform.DropColsWithNaN(df, ['mths_since_last_delinq',
                                          'mths_since_last_record',
                                          'next_payment_date',
                                          'mths_since_last_major_derog'])

data_frame_transform.DropRowsWithNaN(df, ['last_payment_date',
                                          'last_credit_pull_date',
                                          'collections_12_mths_ex_med'])

df_info.PrintNaNFractions(df)


# %% [markdown]
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
predictors = ['funded_amount_inv', 'instalment','total_rec_int','total_rec_prncp','total_payment_inv','total_payment']

mlr_mask = data_frame_transform.DefineMLR2Impute(df, 'funded_amount', predictors)

data_frame_transform.ImputeNaNMLR(df, 'funded_amount', predictors, mlr_mask)

# term has two values, 36 months or 60 months. 
# They have similar frequencies so imputing with the most common is risky.
# We therefore replace NaN with a new category, Uknown.
data_frame_transform.ImputeNaN(df, 'term', 'Unknown')

# Now we've dealt with all the NaN.
df_info.PrintNaNFractions(df)


