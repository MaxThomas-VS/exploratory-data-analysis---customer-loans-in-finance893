import yaml
import sqlalchemy as sqla


def load_credentials(filename):
    with open(filename, 'r') as fn:
        credentials_dict = yaml.safe_load(fn)
    return credentials_dict

class RDSDatabaseConnector():
    
    def __init__(self, filename="credentials.yaml"):
        self.credentials = load_credentials(filename)

    def StartSQLAEngine(self):
        url = sqla.engine.url.create(**db.credentials)
        return sqla.create_engine(url)


if __name__ == '__main__':
    db = RDSDatabaseConnector()
