import yaml

def load_credentials(filename="credentials.yaml"):
    with open(filename, 'r') as fn:
        credentials_dict = yaml.safe_load(fn)
    return credentials_dict

class RDSDatabaseConnector():
    pass

if __name__ == '__main__':
    print(load_credentials())