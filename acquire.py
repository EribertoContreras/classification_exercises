from cgi import test
import numpy as np
import seaborn as sns
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import env
from pydataset import data
import scipy
import os
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

def get_connection(db, user=env.user, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'
def get_titanic_data():
    return pd.read_sql('SELECT * FROM passengers', get_connection('titanic_db'))


def get_iris_data():
    return pd.read_sql('SELECT * FROM species', get_connection('iris_db'))

def get_telco_data():
    return pd.read_sql("""
SELECT * FROM customers
JOIN customer_payments USING(customer_id)
JOIN customer_contracts USING(customer_id)
JOIN customer_subscriptions USING(customer_id)
"""
, get_connection('telco_churn'))



def get_titanic_data():
    filename = "titanic.csv"

    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        # read the SQL query into a dataframe
        df = pd.read_sql('SELECT * FROM passengers', get_connection('titanic_db'))

        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename, index = False)

        # Return the dataframe to the calling code
        return df  


def get_iris_data():
    filename = "iris.csv"

    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        # read the SQL query into a dataframe
        df = pd.read_sql('SELECT * FROM measurements JOIN species USING(species_id)', get_connection('iris_db'))

        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename, index = False)
        #changing it csv because its a csv

        # Return the dataframe to the calling code
        return df  

train, test = train_test_split(df, test_size=.2, random_state=123, stratify=df.species_name)
train, validate = train_test_split(train, test_size=.3, random_state=123, stratify=train.species_name)

# Validate my split.

print(f'train -> {train.shape}')
print(f'validate -> {validate.shape}')
print(f'test -> {test.shape}')

def split_iris_data(df):
    '''
    take in a DataFrame and return train, validate, and test DataFrames; stratify on survived.
    return train, validate, test DataFrames.
    '''
    train_validate, test = train_test_split(df, test_size=.2, random_state=123, stratify=df.species_name)
    train, validate = train_test_split(train_validate, 
                                       test_size=.3, 
                                       random_state=123, 
                                       stratify=train_validate.species_name)
    return train, validate, test


imputer = SimpleImputer(missing_values = None, strategy='most_frequent')





def get_telco_data():
    filename = "telco.csv"

    if os.path.isfile(filename):
        return pd.read_csv(filename)

    else:
        # read the SQL query into a dataframe
        df = pd.read_sql("""
        SELECT * FROM customers
        JOIN contract_types USING(contract_type_id)
        JOIN payment_types USING(payment_type_id)
        JOIN internet_service_types USING(internet_service_type_id);""", get_connection('telco_churn'))

        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename, index = False)
        #changing it csv because its a csv

        # Return the dataframe to the calling code
        return df
