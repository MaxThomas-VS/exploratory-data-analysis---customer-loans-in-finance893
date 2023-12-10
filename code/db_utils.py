import yaml
import sqlalchemy as sqla
import pandas as pd

# TODO: this is just a read yaml function. should rename accordingly to get rid of mention of credentials
def load_credentials(filename):
    '''
    Creates a python dictionary from a YAML file. Specifically, we need credentials to access RDS on AWS.

    Parameters:
    ----------
    filename: str
        some .yaml
    
    Returns:
    --------
    credentials: dict
        credentials for accessing AWS RDS
    '''
    with open(filename, 'r') as fn:
        credentials = yaml.safe_load(fn)
    return credentials

def CloudRDS2csv(table_name='loan_payments'):
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
    table.to_csv( "../data/" + table_name )
    return table

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
        self.credentials = load_credentials(filename)

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
    CloudRDS2csv()
