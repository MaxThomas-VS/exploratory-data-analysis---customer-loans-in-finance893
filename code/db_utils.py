import yaml
import sqlalchemy as sqla
import pandas as pd
import sys

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
    



if __name__ == '__main__':
    try:
        table_name = sys.argv[1]
    except:
        table_name = 'loan_payments'

    CloudRDS2csv(table_name)

    df = load_csv(table_name)

    print(df.info())
    print(df.head())
    
