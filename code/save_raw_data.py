from db_utils import RDSDatabaseConnector
import sys

if __name__ == '__main__':
    try:
        table_name = sys.argv[1]
    except:
        table_name = 'loan_payments'

    raw_data = RDSDatabaseConnector.CloudTable2csv(table_name)

    