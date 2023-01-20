#import libraries and the 'get_connection' function from env
import pandas as pd
import numpy as np
import env
import os
from env import get_connection


def wrangle_zillow():
    '''
    looking for an already existing zillow csv on the local machine
    '''
    if os.path.isfile('zillow.csv'):
        return pd.read_csv('zillow.csv')
    else:
        '''
        if there is no existing csv, then connect to the SQL server and get the information from 
        telco_churn db
        '''
        url = get_connection('zillow')
        '''
        use the query to rename columns too
        '''
        query = '''
                SELECT bathroomcnt as bath_count, bedroomcnt as bed_count, 
                taxvaluedollarcnt as property_value, calculatedfinishedsquarefeet as finished_sq_ft, 
                latitude, longitude
                FROM properties_2017
                JOIN propertylandusetype USING(propertylandusetypeid)
                JOIN predictions_2017 USING(parcelid)
                WHERE propertylandusedesc = 'Single Family Residential' AND transactiondate LIKE '2017%%';
                '''
        
        df = pd.read_sql(query, url)
        '''
        drop null values
        '''
        df=df.dropna()
        '''
        saving the newly queried SQL table to a csv so it
        can be used instead of connecting to the SQL server
        every time I want this info
        '''
        df.to_csv('zillow.csv', index=False)
        return df