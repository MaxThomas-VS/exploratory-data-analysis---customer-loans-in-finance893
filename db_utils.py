import yaml
import sqlalchemy as sqla
import pandas as pd


def load_credentials(filename):
    with open(filename, 'r') as fn:
        credentials_dict = yaml.safe_load(fn)
    return credentials_dict

class RDSDatabaseConnector():
    
    def __init__(self, filename="credentials.yaml"):
        self.credentials = load_credentials(filename)

    def StartSQLAEngine(self):
        url = sqla.engine.url.URL.create(**self.credentials)
        return sqla.create_engine(url)
    
    def CloudData2Table(self, table='loan_payments'):
        engine = self.StartSQLAEngine().connect()
        return pd.read_sql_table(table, engine)



if __name__ == '__main__':
    table = RDSDatabaseConnector().CloudData2Table()
    print(table.columns)
