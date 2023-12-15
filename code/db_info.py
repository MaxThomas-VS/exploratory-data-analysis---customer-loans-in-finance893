# %%
import db_utils as dbu

# %%
# Get loan_payments table
table_name = 'loan_payments'
dbu.CloudRDS2csv(table_name)

transform = dbu.DataTransform(table_name)
df = transform.df

# %%
# Do initial clean up
transform.DropOnly1Value()

categorical_columns = ['grade', 'sub_grade', 'employment_length', 'home_ownership','verification_status','loan_status','payment_plan','purpose','term', 'collections_12_mths_ex_med','inc_last_6_mnths','inq_last_6mths']
transform.MakeCategorical(categorical_columns)

datenum_columns = ['issue_date', 'earliest_credit_line', 'last_payment_date', 'next_payment_date', 'last_credit_pull_date']
transform.Dates2Datetimes(datenum_columns)


# %%
# Get info about dataframe
get_info = dbu.DataFrameInfo(df)
get_info.DescribeDataFrame()

# %% 
# Deal with missing values
transform_df = dbu.DataFrameTransform(df)
plotter = dbu.Plotter(df)
plotter.InspectNaN()

#plotter.PairPlot(['loan_amount','annual_inc'])

# %%
# Vars with missing:
# funded_amount, term, int_rate, employment_length, mths_since_last_delinq, mths_since_last_record, last_payment_date, next_payment_date, last_credit_pull_date, collections_12_mths_ex_med, mths_since_last_major_derog
get_info.PrintColumnInfo('loan_amount')
plotter.Histogram('loan_amount')




# %%
# Drop for high nan count: mths_since_last_delinq, mths_since_last_record, mths_since_last_major_derog, 'next_payment_date'
transform_df.DropColsWithNaN(['mths_since_last_delinq', 'mths_since_last_record', 'mths_since_last_major_derog', 'next_payment_date'])
get_info.DescribeDataFrame()

# %%
# Impute with 0 for collections_12_mths_ex_med, which is by far most common
transform_df.ImputeNaN('collections_12_mths_ex_med', 0)

# Impute with 36 months for term
# TODO: revise this maybe, as it's an uncomfortable choice
transform_df.ImputeNaN('term','36 months')

# %%
# Drop rows with missing NaN from int_rate, funded amount, and employment_length, which have a large std and are important variables
# Drop rows with missing NaN for last_credit_pull_date, which is missing very few values
transform_df.DropRowsWithNaN(['int_rate','funded_amount','last_credit_pull_date','employment_length'])


# %% 
# Tidy up and print results and summary figure
df.reset_index(drop=True, inplace=True)
df.drop(columns=['Unnamed: 0'], inplace=True)
get_info.DescribeDataFrame()



# %%
isnumeric = get_info.IsNumeric()
#plotter.PairPlot(isnumeric)

# %%
plotter.CorrelationHeatmap(isnumeric)

# %%
skew = df.skew(numeric_only=True) 
very_skewed = list(skew.index[abs(skew)>1])
moderately_skewed = list(skew.index[(abs(skew)>0.5) & (abs(skew)<=1)])

#df.kurtosis(numeric_only=True)


# %%
# Correct Skew for moderately skewed data
# note: log transform performed poorly
# 
df2 = df.copy() # stop working on df

plotter2 = dbu.Plotter(df2)
for col in moderately_skewed:
    fig = dbu.plt.figure()
    plotter2.Histogram(col)
    
    fig = dbu.plt.figure()
    plotter2.Histogram(col, transform='yeojohnson')

# %%
# These are much less skewed, so we overwrite them in df2
for col in moderately_skewed:
    df2[col] = dbu.stats.yeojohnson(df2[col])[0]

# %%
# Correct Skew for severely skewed data
# yj corrects: annual_inc, open_accounts, total_payment, total_payment_inv, total_rec_prncp, total_rec_int
# several variables appear to be multimodal: deling_2yrs, out_prncp_inv, out_prncp, total_rec_late_fee, recoveries, collection_recovery_fee, last_payment_amount
# memberid and id are ignored as these are presumably unique identifiers

plotter2 = dbu.Plotter(df2)
for col in very_skewed:
    fig = dbu.plt.figure()
    plotter2.Histogram(col)
    
    fig = dbu.plt.figure()
    plotter2.Histogram(col, transform='yeojohnson')

for col in ['annual_inc', 'open_accounts', 'total_payment', 'total_payment_inv', 'total_rec_prncp', 'total_rec_int']:
    df2[col] = dbu.stats.yeojohnson(df2[col])[0]

# %%
# Save df2
df2.to_csv('../data/loan_repayments_corrected.csv')
