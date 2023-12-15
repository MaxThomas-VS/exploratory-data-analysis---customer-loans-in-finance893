from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import missingno as msno
import pandas as pd
from scipy import stats
import seaborn as sns
import sqlalchemy as sqla
import sys
import yaml


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

    def DropOnly1Value(self): # remove columns with only one value as they effectively contain no information
        to_drop = []
        for SeriesName, series in df.items():
            if len(series.unique()) == 1:
                to_drop.append(SeriesName)
        df.drop(labels=to_drop, axis=1, inplace=True)
        
    def MakeCategorical(self, columns):
        for column in columns:
            df[column] = df[column].astype('category') 

    def DropSpecificColumns(self, columns):
        df.drop(labels=columns, axis=1, inplace=True)

    def CorrectEmploymentLength(self):
        df.employment_length = df.employment_length.replace(' year', '')
        df.employment_length = df.employment_length.replace(' years', '')

    def Dates2Datetimes(self, columns):
        for column in columns:
            df[column] = pd.to_datetime(df[column], format='%b-%Y')

    def CorrectTerm(self):
        for ix, ixs in enumerate(df.term.str.split(' ')):
            try:
                df.term[ix] = ixs[0]
            except:
                df.term[ix] = ixs

class DataFrameInfo():

    def GetNaNFraction(self, column):
        return df[column].isna().sum() / len(df[column])

    def IsNumeric(self):
        isnumeric = []
        for column in df.columns:
            if df[column].dtype.kind in 'biufc':
                isnumeric.append(column)
        return isnumeric

    def GetColumnInfo(self, column):
        description = df[column].describe()
        column_info = {'name': column,
                       'dtype': str(df[column].dtype),
                       'length': len(df[column]),
                       'NaN_pct': 100 * self.GetNaNFraction(column)}
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
    
    def PrintColumnInfo(self, column):
        column_info = self.GetColumnInfo(column)
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

    def DescribeDataFrame(self, filename='initial_data_description.txt'):
        print(df.info())
        print('==============================')
        print(df.describe())
        print('==============================')
        for col in df.columns:
            self.PrintColumnInfo(col)

class DataFrameTransform():

    def GetNaNFraction(self, column):
        return df[column].isna().sum() / len(df[column])

    def ImputeNaN(self, column, method='median'):
        if method=='median':
            impute_value = df[column].median()
        elif method=='mean':
            impute_value = df[column].mean()
        else:
            impute_value = method
        df[column] = df[column].fillna(impute_value)

    def DropRowsWithNaN(self, columns):
        df.dropna(axis=0, subset=columns, inplace=True)

    def DropColsWithNaN(self, columns):
        df.drop(columns=columns, inplace=True)

    def RemoveOutliers(self, column):
        pass

class Plotter():

    def __init__(self, df):
        df = df

    def Histogram(self, column, transform=False):
        data = df[column].copy()
        if transform == 'log':
            data = data.map(lambda i: np.log(i) if i > 0 else 0)
        elif transform == 'boxcox':
            data = stats.boxcox(data)
            data = pd.Series(data[0])
        elif transform == 'yeojohnson':
            data = stats.yeojohnson(data)
            data = pd.Series(data[0])          
        t = sns.histplot(data, label="Skewness: %.2f"%(data.skew()), kde=True )
        t.legend()

    def PairPlot(self, columns):
        sns.pairplot(df[columns])

    def CorrelationHeatmap(self, columns):
        corr = df[columns].corr()
        mask = np.zeros_like(corr, dtype=np.bool_)
        mask[np.triu_indices_from(mask)] = True
        sns.heatmap(corr, square=True, linewidths=.5, annot=False, cmap='seismic', mask=mask)
        plt.yticks(rotation=0)
        plt.title('Correlation Matrix of all Numerical Variables')

    def InspectNaN(self):
        msno.bar(df)
        msno.matrix(df)
        msno.heatmap(df)

    