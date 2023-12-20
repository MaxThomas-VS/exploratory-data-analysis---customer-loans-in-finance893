from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import missingno as msno
import pandas as pd
from scipy import stats
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import sqlalchemy as sqla
import sys
import yaml
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing




def read_yaml(filename):
    '''
    Creates a python dictionary from a YAML file. Here, we need credentials to access RDS on AWS.

    Parameters:
    ----------
    filename: str
        some .yaml
    
    Returns:
    --------
    yaml_as_dict: dict
        yaml_as_dict for accessing AWS RDS
    '''
    with open(filename, 'r') as fn:
        yaml_as_dict = yaml.safe_load(fn)
    return yaml_as_dict

def load_csv(table_name):
    '''
    Loads csv for given table to pandas dataframe.
    '''
    return pd.read_csv('../data/' + table_name + '.csv')

class RDSDatabaseConnector():
    '''
    Class to connect to relational database (RDS) stored on AWS.

    Parameters:
    ----------
    filename: str
        name of YAML file to gather credential from, extension included
    
    Attributes:
    ----------
    credentials: dict
        credentials for acessing RDS

    Methods:
    -------
    StartSQLAEngine()
        Creates engine to connect with RDS.

    CloudData2Table(table)
        Connects with RDS and returns a given <table>.
    '''
    def __init__(self, filename="../setup/credentials.yaml"):
        self.credentials = read_yaml(filename)

    def StartSQLAEngine(self):
        '''
        Creates engine to connect to RDS.
        '''
        url = sqla.engine.url.URL.create(**self.credentials) # converts credentials into url
        return sqla.create_engine(url)
    
    def CloudData2Table(self, table='loan_payments'):
        '''
        Connects to engine and returns a given table as a pandas dataframe.
        '''
        engine = self.StartSQLAEngine().connect()
        return pd.read_sql_table(table, engine)
    
    def CloudTable2csv(table_name):
        '''
        Creates a pandas dataframe from a table in the RDS and saves to a csv in ../data/.

        Parameters:
        ----------
        table_name: str
            name of table to be accessed. The saved csv is at ../data/<table_name>.csv
        
        Returns:
        --------
        table: pandas.DataFrame
            slected table as dataframe
        '''
        db = RDSDatabaseConnector()
        table = db.CloudData2Table(table_name)
        table.to_csv( "../data/" + table_name + '-raw.csv' )
        return table
    
class DataTransform():

    def DropOnly1Value(self, df): # remove columns with only one value as they effectively contain no information
        to_drop = []
        for SeriesName, series in df.items():
            if len(series.unique()) == 1:
                to_drop.append(SeriesName)
        df.drop(labels=to_drop, axis=1, inplace=True)
        
    def MakeCategorical(self, df, columns):
        for column in columns:
            df[column] = df[column].astype('category') 

    def DropSpecificColumns(self, df, columns):
        df.drop(labels=columns, axis=1, inplace=True)

    def CorrectEmploymentLength(self, df):
        df.employment_length = df.employment_length.replace(' year', '')
        df.employment_length = df.employment_length.replace(' years', '')

    def Dates2Datetimes(self, df, columns):
        for column in columns:
            df[column] = pd.to_datetime(df[column], format='%b-%Y')



class DataFrameInfo():

    def GetNaNFraction(self, df, column):
        return df[column].isna().sum() / len(df[column])
    
    def PrintNaNFractions(self, df):
        some_nan = False
        for col in df.columns:
            nan_frac = self.GetNaNFraction(df, col)
            if nan_frac > 0:
                some_nan = True
                print('%s has %s %% NaN.' % (col, 100*nan_frac) )
        if not some_nan:
            print('No NaN in dataset.')
        

    def IsNumeric(self, df):
        isnumeric = []
        for column in df.columns:
            if df[column].dtype.kind in 'biufc':
                isnumeric.append(column)
        return isnumeric

    def GetColumnInfo(self, df, column):
        description = df[column].describe()
        column_info = {'name': column,
                       'dtype': str(df[column].dtype),
                       'length': len(df[column]),
                       'NaN_pct': 100 * self.GetNaNFraction(df, column)}
        if column_info['dtype'] == 'category':
            column_info['unique'] = description['unique']
            column_info['top'] = description['top']
            column_info['top_fraction'] = 100 * description['freq'] / column_info['length']
        else:
            column_info['mean'] = description['mean']
            column_info['min']  = description['min']
            column_info['max']  = description['max']
            column_info['25%'] = description['25%']
            column_info['50%'] = description['50%']
            column_info['75%'] = description['75%']
            column_info['std'] = df[column].std()
        return column_info
    
    def PrintColumnInfo(self, df, column):
        column_info = self.GetColumnInfo(df, column)
        print('---------------------')
        print('%s is a %s with %s data, of which %s %% are NaN.' % 
              (column_info['name'], column_info['dtype'], column_info['length'], column_info['NaN_pct']))
        if column_info['dtype'] == 'category':
            print('There are %s categories, of which %s is the most common at %s %% of the total.' %
                  (column_info['unique'], column_info['top'], column_info['top_fraction']))
        else:
            print('The range of the data is %s to %s.' % (column_info['min'], column_info['max']))
            print('The mean is %s with a standard deviation of %s.' % (column_info['mean'], column_info['std']))
            print('The median is %s, and the 25 and 75 %% centiles are %s and %s.' % 
                  (column_info['50%'], column_info['25%'], column_info['75%']))
        print('---------------------')

    def DescribeDataFrame(self, df, filename='initial_data_description.txt'):
        print(df.info())
        print('==============================')
        print(df.describe())
        print('==============================')
        for col in df.columns:
            self.PrintColumnInfo(df, col)

class DataFrameTransform():

    def GetNaNFraction(self, df, column):
        return df[column].isna().sum() / len(df[column])

    def ImputeNaN(self, df, column, method='median'):
        if method=='median':
            impute_value = df[column].median()
        elif method=='mean':
            impute_value = df[column].mean()
        else:
            if df[column].dtype == 'category':
                df[column] = df[column].cat.add_categories(method)
                impute_value = method
            else:
                impute_value = method

        df[column] = df[column].fillna(impute_value)


    def DefineMLR2Impute(self, df, column, predictors):
        df_predictors = df[predictors]
        df_response = df[column]
        build_mask = df_response.notna()
        Xs = df_predictors.loc[build_mask]
        Ys = df_response.loc[build_mask]

        X_train, X_test, Y_train, Y_test = train_test_split(Xs, Ys, test_size = 0.2, random_state = 100)

        mlr = LinearRegression().fit(X_train, Y_train)
        predicted = mlr.predict(X_test)

        plt.figure()
        plt.scatter(np.array(Y_test), np.array(predicted), s=0.1)
        plt.title('MLR performance for ' + column)

        return mlr, build_mask
    
    def ImputeNaNMLR(self, df, column, predictors, mlr=None):
        if mlr is None:
            mlr, mask = self.DefineMLR2Impute(df, column, predictors)
        else:
            mlr, mask = mlr[0], mlr[1]
        df[column].loc[~mask] = mlr.predict(df[predictors].loc[~mask])

                    
    def DropRowsWithNaN(self, df, columns):
        df.dropna(axis=0, subset=columns, inplace=True)

    def DropColsWithNaN(self, df, columns):
        df.drop(columns=columns, inplace=True)

    def RemoveOutliers(self, df, column):
        pass

    def TransformColumn(self, df, col, transform='log'):
        var_to_trans = df[col].copy()
        if transform == 'log':
            trans_var = var_to_trans.map(lambda ii: np.log(ii) if ii > 0 else 0)
        elif transform == 'boxcox':
            try:
                trans_var = pd.Series(stats.boxcox(var_to_trans)[0]).rename(col, inplace=True)
            except ValueError:
                print('Box-Cox fails for %s as some data are negative.' % (col))
                print('Returning original data.')
                trans_var = var_to_trans
        elif transform == 'yeojohnson':
            trans_var = pd.Series(stats.yeojohnson(var_to_trans)[0]).rename(col, inplace=True)
        elif transform == 'No transform':
            trans_var = var_to_trans
        return trans_var

    def ImputeTerm(self, df):
        le = preprocessing.LabelEncoder()
        impute_col = 'term'
        impute_col_encoded = 'term_encoded'

        le.fit(df.loc[:, impute_col])

        df[impute_col_encoded] = le.transform(df[impute_col])
        df[impute_col_encoded] = df[impute_col_encoded].map(lambda x: np.nan if x == 2 else x)

        imputer = IterativeImputer(random_state=1)
        df_train = df[['loan_amount','instalment','int_rate',impute_col_encoded]]
        imputer.fit(df_train)

        df_imputed = imputer.transform(df_train)
        new_term = df_imputed[:,-1].round().astype(int)
        #new_term = new_term.map(lambda x: 0 if x < 0.5 else 1 )

        df[impute_col_encoded] = new_term
        df[impute_col_encoded] = df[impute_col_encoded].map(lambda x: 0 if x < 0.5 else 1 )

        term_imputed = list(le.inverse_transform(df[impute_col_encoded]))

        df[impute_col] = term_imputed
    

class Plotter():

    def Histogram(self, df, column, transform=False, ax=False):
        data = df[column].copy()
        if transform == 'log':
            data = data.map(lambda i: np.log(i) if i > 0 else 0)
        elif transform == 'boxcox':
            data = stats.boxcox(data)
            data = pd.Series(data[0])
        elif transform == 'yeojohnson':
            data = stats.yeojohnson(data)
            data = pd.Series(data[0])      
        if not ax:    
            t = sns.histplot(data, label="Skewness: %.2f"%(data.skew()), kde=True )
            t.legend()
        else:
            sns.histplot(data, label="Skewness: %.2f"%(data.skew()), kde=True, ax=ax)
            ax.legend()

    def TransformTest(self, df, col):

        fig, axes = plt.subplots(1,4, figsize=(10, 7))
        transformers = ['No transform', 'log', 'boxcox', 'yeojohnson']
        tc = DataFrameTransform()
        pl = Plotter()
        for itrns, trns in enumerate(transformers):
            var2test = tc.TransformColumn(df=df, col=col, transform=trns).to_frame()
            pl.Histogram(var2test, col, ax=axes[itrns])
            axes[itrns].title.set_text(trns)
            #sns.histplot(var2test[col], ax=axes[icol])

        fig.tight_layout()
        

    def PairPlot(self, df, columns):
        sns.pairplot(df[columns])

    def CorrelationHeatmap(self, df, columns):
        corr = df[columns].corr()
        mask = np.zeros_like(corr, dtype=np.bool_)
        mask[np.triu_indices_from(mask)] = True
        plt.figure(figsize=(15,15))
        sns.heatmap(corr, square=True, linewidths=.5, annot=True, cmap='seismic', vmin=-1, vmax=1, mask=mask)
        plt.yticks(rotation=0)
        plt.title('Correlation Matrix')

    def InspectNaN(self, df, plots=['bar', 'matrix', 'heatmap']):
        if 'bar' in plots:
            msno.bar(df)
        if 'matrix' in plots:
            msno.matrix(df)
        if 'heatmap' in plots:
            msno.heatmap(df)

    def CheckOutlierRemoval(self, df, df_test, columns):
        to_check_outliers = columns
        for col in to_check_outliers:
            fig, axes = plt.subplots(2,1, figsize=(5,10))
            self.Histogram(df, col, ax=axes[0])
            self.Histogram(df_test, col, ax=axes[0])
            boxplot_data = {'Raw': df[col], 'Outliers removed': df_test[col]}
            axes[1].boxplot(boxplot_data.values(),
                        flierprops={'marker': 'x', 'markersize': 1, 'markerfacecolor': 'r'})
            axes[1].set_xticklabels(boxplot_data.keys())
            axes[1].set_ylabel(col)
            print('==================================')


    