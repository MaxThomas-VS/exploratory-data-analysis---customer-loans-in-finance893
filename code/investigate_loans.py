# %%
import db_utils as dbu
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing

import pandas as pd
import numpy as np
import seaborn as sns

def get_potential_value(df):
    # P: principle [GBP], r: interest rate, t: term [years]
    P, r, t = df['loan_amount'], 1+df['int_rate']/100, df['term_numeric']/12 
    return ( P * ( r ** t ) ).sum()

def get_loss_breakdown(df, loan_status):
    loss_breakdown = {}
    if loan_status:
        df2 = df.loc[df['loan_status-simple']==loan_status]
    else:
        df2 = df
    loss_breakdown['total_loaned'] = df2['funded_amount'].sum()
    loss_breakdown['total_recovered'] = df2['total_payment'].sum()
    loss_breakdown['potential_value'] = get_potential_value(df2)
    if loan_status == 'Charged Off':
        loss_breakdown['extra_recoveries'] = df2['recoveries'].sum()
    else:
        loss_breakdown['extra_recoveries'] = 0
    loss_breakdown['total_lost'] = loss_breakdown['total_loaned'] - (loss_breakdown['total_recovered'] + loss_breakdown['extra_recoveries'])
    loss_breakdown['potential_lost'] = loss_breakdown['potential_value'] - (loss_breakdown['total_recovered'] + loss_breakdown['extra_recoveries'])
    return loss_breakdown

def label_function(val):
    return f'{val / 100 * len(df):.0f}\n{val:.0f}%'

def get_pct_loanstatus(df, status):
    N = len(df)
    n = df['loan_status-simple'].value_counts()[status]
    return 100 * n / N
    
def x_pct_of_y(x, y, df):
    if df is None:
        return 100 * x / y
    else:
        return 100 * df[x].sum() / df[y].sum()

def pct_status(df, column, value, status):
    df_subset = df.loc[df[column]==value]
    len_df = len(df_subset)
    num_status = df_subset['loan_status-simple'].value_counts()[status]
    return 100 * num_status / len_df

def pct_status_series(df, column, status):
    values = []
    cats = []
    for cg in df[column].cat.categories:
        values.append(pct_status(df,column,cg,status))
        cats.append(cg)
    return pd.Series(values, index=cats)

# %%
df = pd.read_pickle('../data/loan_payments-clean.pkl')

insights = {}

# %% 
transformer = dbu.DataFrameTransform()
transformer.SimpleLoanStatus(df)

# %%
# What percentage of loans are recovered against investor funding and total funding?
# What percentage would be recovered up to 6 months in future?


insights = {
        'recovered_total': x_pct_of_y('total_payment', 'funded_amount', df),
        'recovered_total_inv': x_pct_of_y('total_payment_inv', 'funded_amount_inv', df)
           }

print('Over the entire dataset, the recovered total is:')
print('> %s %% of total funding' % (round(insights['recovered_total'], 2)))
print('> %s %% of investor funding' % (round(insights['recovered_total_inv'], 2)))
print('')



df['total_payment-est+6'] = np.where(df['loan_status']=='Current', df['total_payment']+df['instalment']*6, df['total_payment'])

insights['recovered_total-est+6'] = x_pct_of_y('total_payment-est+6', 'funded_amount', df)
print('Over the next 6 months, repayment of current loans would generate %s million GBP in revenue.' % (df['total_payment-est+6'].sum() * 10**-6))
print('This would increase recovery to %s %% of the total commited.' % (insights['recovered_total-est+6']))


# %%
# Charged off
loss_breakdown = {}
df['loan_status'].value_counts()
pct_charged_off = get_pct_loanstatus(df, 'Charged Off')
#total_paid_to_charged_off = df['total_payment'].loc[df['loan_status']=='Charged Off'].sum()
#total_funded_charged_off = df['funded_amount'].loc[df['loan_status']=='Charged Off'].sum()
#insights['recovered_from_charged_off'] = x_pct_of_y(total_paid_to_charged_off, total_funded_charged_off, df=None)

loss_breakdown['Charged Off'] = get_loss_breakdown(df, 'Charged Off')
#co_loaned, co_recovered, co_value = get_loss_breakdown(df, 'Charged Off')
#insights['recovered_from_charged_off'] = 100 * co_recovered / co_loaned

insights['recovered_from_charged_off'] = \
    100 * (1 - loss_breakdown['Charged Off']['total_lost']/loss_breakdown['Charged Off']['total_loaned']) 
post_recovery_charged_off = 100 * loss_breakdown['Charged Off']['extra_recoveries']/loss_breakdown['Charged Off']['total_recovered']

print('%s %% of charged off loans were recovered.' % (insights['recovered_from_charged_off']))
print('%s %% of charged off recoveries were recieved after closure of the loan.' % (post_recovery_charged_off))
plt.bar(x=insights.keys(), height=insights.values())


#%%
#df_co = df.loc[df['loan_status']=='Charged Off']
#df_co['term_numeric'] = [36 if ix == '36 months' else 60 for ix in df_co['term']]
#potential_value = df_co['loan_amount'] * (1+df_co['int_rate']/100) ** (df_co['term_numeric']/12)

#co_potential_lost = co_value - co_recovered
#co_post_recovery = df.loc[df['loan_status']=='Charged Off']['recoveries'].sum()
#co_lost = co_loaned - (co_recovered + co_post_recovery)

plt.bar(x=loss_breakdown['Charged Off'].keys(),
        height=loss_breakdown['Charged Off'].values())

#plt.pie([co_lost_value, co_recovered, co_post_recovery],
#        labels=['Charged off', 'Repayed', 'Recovered'])
# %%

fig, ax = plt.subplots(ncols=1, figsize=(5, 5))

df.groupby('loan_status-simple').size().plot(kind='pie', autopct=label_function, textprops={'fontsize': 10},
                                  colors=['tomato', 'gold', 'skyblue', 'green', 'magenta'], ax=ax)

## %%
# What percentage do users' in this bracket currently represent as a percentage of all loans?
pct_late = get_pct_loanstatus(df, 'Late')
print('%s %% of customers are late on loan repayments.' % (round(pct_late,1)))
# Calculate the total amount of customers in this bracket and how much loss the company would incur their status was changed to Charged Off. 
loss_breakdown['Late'] = get_loss_breakdown(df, 'Late')

print('The total loss if all those in arrears were charged off would be %s Million GBP.' % (round(loss_breakdown['Late']['total_lost'],2) / 10**6))
# What is the projected loss of these loans if the customer were to finish the full loans term?
print('The loss relative to the value of full repayment for all those in arrears were charged off would be %s Million GBP.' % (round(loss_breakdown['Late']['potential_lost'],2) / 10**6))


#%%
#If customers late on payments converted to Charged Off, what percentage of total expected revenue do these customers and the customers who have already defaulted on their loan represent?
df_co_late = df.loc[ (df['loan_status-simple']=='Charged Off') | 
                     (df['loan_status-simple']=='Late') | 
                     (df['loan_status-simple']=='Default')]

loss_breakdown['Late->Charged Off'] = get_loss_breakdown(df_co_late, None)

loss_breakdown['Current'] = get_loss_breakdown(df, 'Current')

print('The potential revenue from all outstanding loans is %s Million GBP.' % (loss_breakdown['Current']['potential_lost'] * 10**-6))
print('If all those in arrears were charged off, the total charged off revenue would be %s Million GBP.' % (loss_breakdown['Late->Charged Off']['potential_lost'] * 10**-6))
print('This is %s %% of the total.' % (100 * loss_breakdown['Late->Charged Off']['potential_lost'] / loss_breakdown['Current']['potential_lost']))


# %%
# for given value in column, check percentage of charged off
data_frame_info = dbu.DataFrameInfo()
iscat = data_frame_info.IsCategorical(df)

cat_pct = {}
for cv in iscat:
    cat_pct[cv] = pct_status_series(df, cv, 'Charged Off')
    sns.barplot(x=cat_pct[cv].index, y=cat_pct[cv].values, order=cat_pct[cv].sort_values().index)

#%%
grade = pct_status_series(df, 'grade', 'Charged Off')
sns.barplot(x=grade.index, y=grade.values, order=grade.sort_values().index)

plt.figure()
purpose = pct_status_series(df, 'purpose', 'Charged Off')
sns.barplot(x=purpose.index, y=purpose.values, order=purpose.sort_values().index)
plt.xticks(rotation=45, ha='right')

plt.figure()

home_ownership = pct_status_series(df, 'home_ownership', 'Charged Off')
sns.barplot(x=home_ownership.index, y=home_ownership.values, order=home_ownership.sort_values().index)
plt.xticks(rotation=45, ha='right')


# %%
variable = 'dti'
fig, ax = plt.subplots(1,1)
colours = ['black','green','red','blue']
statuss = ['Fully Paid', 'Current', 'Charged Off', 'Late']
for status, colour in zip(statuss, colours):
    to_plot = df[variable].loc[df['loan_status-simple']==status]
    median = to_plot.median()
    label = status + ': median = ' + str(np.round(median,2))
    sns.kdeplot(to_plot, label=label, color=colour)
    plt.axvline(median, color=colour, linestyle='dashed', linewidth=1)
plt.legend(loc='upper right')
    #sns.histplot(to_plot, kde=True, element='step', fill=False, stat='density', ax=ax, hue='white')
# %%
