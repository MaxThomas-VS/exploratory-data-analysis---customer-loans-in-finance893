from datetime import datetime
import pandas as pd
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

def CloudRDS2csv(table_name):
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
    table.to_csv( "../data/" + table_name + '.csv' )
    return table

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
    def __init__(self, filename="credentials.yaml"):
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
    
class DataTransform():

    def __init__(self, table_name):
        self.df = load_csv(table_name)

    def DropOnly1Value(self): # remove columns with only one value as they effectively contain no information
        to_drop = []
        for SeriesName, series in self.df.items():
            if len(series.unique()) == 1:
                to_drop.append(SeriesName)
        self.df.drop(labels=to_drop, axis=1, inplace=True)
        
    def MakeCategorical(self, columns):
        for column in columns:
            self.df[column] = self.df[column].astype('category') 

    def DropSpecificColumns(self, columns):
        self.df.drop(labels=columns, axis=1, inplace=True)

    def DropRowsWithNaN(self, columns):
        self.df.dropna(axis=0, subset=columns, inplace=True)
    
    def InputeValues(self, method='mean'):
        pass

    def CorrectEmploymentLength(self):
        self.df.employment_length = self.df.employment_length.replace(' year', '')
        self.df.employment_length = self.df.employment_length.replace(' years', '')

    def Dates2Datetimes(self, columns):
        for column in columns:
            self.df[column] = pd.to_datetime(self.df[column], format='%b-%Y')

    def CorrectTerm(self):
        for ix, ixs in enumerate(self.df.term.str.split(' ')):
            try:
                self.df.term[ix] = ixs[0]
            except:
                self.df.term[ix] = ixs

class DataFrameInfo():
    
    def __init__(self, df):
        self.df = df

    def GetColumnInfo(self, column):
        description = self.df[column].describe()
        info = self.df[column].info()
        column_info = {'name': column,
                       'dtype': str(self.df[column].dtype),
                       'length': len(self.df[column]),
                       'NaN_fraction': self.df[column].isna().sum() / len(self.df[column])}
        if column_info['dtype'] == 'category':
            column_info['unique'] = description['unique']
            column_info['top'] = description['top']
            column_info['top_fraction'] = column_info['top'] / column_info['length']
        else:
            pass






if __name__ == '__main__':
    try:
        table_name = sys.argv[1]
    except:
        table_name = 'loan_payments'

    CloudRDS2csv(table_name)

    raw_data = DataTransform(table_name)

    print(raw_data.df.info())
    print(raw_data.df.head())

    raw_data.DropOnly1Value()

    categorical_columns = ['grade', 'sub_grade', 'employment_length', 'home_ownership','verification_status','loan_status','payment_plan','purpose']
    raw_data.MakeCategorical(categorical_columns)
    
    datenum_columns = ['issue_date', 'earliest_credit_line', 'last_payment_date', 'next_payment_date', 'last_credit_pull_date']
    raw_data.Dates2Datetimes(datenum_columns)

    raw_data.CorrectTerm()
