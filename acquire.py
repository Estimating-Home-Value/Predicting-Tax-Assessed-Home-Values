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
        SELECT p.calculatedfinishedsquarefeet, p.bathroomcnt, p.bedroomcnt, p.taxvaluedollarcnt
        FROM properties_2017 AS p
        JOIN predictions_2017 AS pr USING (parcelid) 
        WHERE p.propertylandusetypeid IN (261, 262, 263, 264, 266, 268, 273, 275, 276, 279)
        AND pr.transactiondate between '2017-05-01' AND '2017-06-30'
        '''
        url = get_connection('zillow')
        zillow = pd.read_sql(query, url)
        zillow.to_csv('zillow.csv')
    zillow = pd.read_csv('zillow.csv')
    zillow.drop(columns=['Unnamed: 0'], inplace=True)
#   zillow['County'] = zillow.apply(lambda row: add_county_column(row), axis = 1)
    return zillow

def acquire_only(query):
    url = get_connection('zillow')
    zillow = pd.read_sql(query, url, index_col='id')
#   zillow['County'] = zillow.apply(lambda row: add_county_column(row), axis = 1)
    return zillow

def acquire_cache_tax_data():
    if not path.isfile('zillow_tax_rate.csv'):
        query = '''
        SELECT p.fips, ROUND((p.taxamount / p.taxvaluedollarcnt) * 100, 2) AS taxrate
        FROM properties_2017 AS p
        JOIN predictions_2017 AS pr USING (parcelid) 
        WHERE p.propertylandusetypeid IN (261, 262, 263, 264, 266, 268, 273, 275, 276, 279)
        AND pr.transactiondate between '2017-05-01' AND '2017-06-30'
        '''
        url = get_connection('zillow')
        zillow = pd.read_sql(query, url)
        zillow.to_csv('zillow_tax_rate.csv')
    zillow = pd.read_csv('zillow_tax_rate.csv')
    zillow.drop(columns=['Unnamed: 0'], inplace=True)
    zillow['County'] = zillow.apply(lambda row: add_county_column(row), axis = 1)
    zillow.drop(columns=['fips'], inplace=True)
    return zillow