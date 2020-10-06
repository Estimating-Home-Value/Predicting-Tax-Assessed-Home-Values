import env
import pandas as pd 
from os import path 

def get_connection(database):
    '''
    Database: string; name of database that the url is being created for
    '''
    return f'mysql+pymysql://{env.user}:{env.password}@{env.host}/{database}'

def add_county_column(row):
    if row['fips'] == 6037:
        return 'Los Angeles'
    elif row['fips'] == 6059:
        return 'Orange'
    elif row['fips'] == 6111:
        return 'Ventura'

def acquire_cache_data():
    if not path.isfile('zillow.csv'):
        query = '''
        SELECT * FROM properties_2017
        '''
        url = get_connection('zillow')
        zillow = pd.read_sql(query, url, index_col='id')
        zillow.to_csv('zillow.csv')
    zillow = pd.read_csv('zillow.csv')
    zillow['County'] = zillow.apply(lambda row: add_county_column(row), axis = 1)
    return zillow

def acquire_only(query):
    url = get_connection('zillow')
    zillow = pd.read_sql(query, url, index_col='id')
    zillow['County'] = zillow.apply(lambda row: add_county_column(row), axis = 1)
    return zillow