import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer, RobustScaler, MinMaxScaler

# Each scaler function should create the object, fit and transform both train and test.
# They should return the scaler, train dataframe scaled, test dataframe scaled. 
# Be sure your indices represent the original indices from train/test, 
# as those represent the indices from the original dataframe. 
# Be sure to set a random state where applicable for reproducibility!

def split_my_data(df, pct=0.10):
    '''
    This divides a dataframe into train, validate, and test sets. 

    Parameters - (df, pct=0.10)
    df = dataframe you wish to split
    pct = size of the test set, 1/2 of size of the validate set

    Returns three dataframes (train, validate, test)
    '''
    train_validate, test = train_test_split(df, test_size=pct, random_state = 123)
    train, validate = train_test_split(train_validate, test_size=pct*2, random_state = 123)
    return train, validate, test

def split_stratify_my_data(df, strat, pct=0.10):
    '''
    This divides a dataframe into train, validate, and test sets straifying on the selected feature

    Parameters - (df, pct=0.10, stratify)
    df = dataframe you wish to split
    pct = size of the test set, 1/2 of size of the validate set
    stratify = a string of the column name of the feature you wish to stratify on

    Returns three dataframes (train, validate, test)
    '''
    train_validate, test = train_test_split(df, test_size=pct, random_state = 123, stratify=df[strat])
    train, validate = train_test_split(train_validate, test_size=pct*2, random_state = 123, stratify=train_validate[strat])
    return train, validate, test

def standard_scaler(train, validate, test):
    '''
    Accepts three dataframes and applies a standard scaler to convert values in each dataframe
    based on the mean and standard deviation of each dataframe respectfully. 
    Columns containing object data types are dropped, as strings cannot be directly scaled.

    Parameters (train, validate, test) = three dataframes being scaled
    
    Returns (scaler, train_scaled, validate_scaled, test_scaled)
    '''
    # Remove columns with object data types from each dataframe
    train = train.select_dtypes(exclude=['object'])
    validate = validate.select_dtypes(exclude=['object'])
    test = test.select_dtypes(exclude=['object'])
    # Fit the scaler to the train dataframe
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True).fit(train)
    # Transform the scaler onto the train, validate, and test dataframes
    train_scaled = pd.DataFrame(scaler.transform(train), columns=train.columns.values).set_index([train.index.values])
    validate_scaled = pd.DataFrame(scaler.transform(validate), columns=validate.columns.values).set_index([validate.index.values])
    test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns.values).set_index([test.index.values])
    return scaler, train_scaled, validate_scaled, test_scaled

def scale_inverse(scaler, train_scaled, validate_scaled, test_scaled):
    '''
    Takes in three dataframes and reverts them back to their unscaled values

    Parameters (scaler, train_scaled, validate_scaled, test_scaled)
    scaler = the scaler you with to use to transform scaled values to unscaled values with. Presumably the scaler used to transform the values originally. 
    train_scaled, validate_scaled, test_scaled = the dataframes you wish to revert to unscaled values

    Returns train_unscaled, validated_unscaled, test_unscaled
    '''
    train_unscaled = pd.DataFrame(scaler.inverse_transform(train_scaled), columns=train_scaled.columns.values).set_index([train_scaled.index.values])
    validate_unscaled = pd.DataFrame(scaler.inverse_transform(validate_scaled), columns=validate_scaled.columns.values).set_index([validate_scaled.index.values])
    test_unscaled = pd.DataFrame(scaler.inverse_transform(test_scaled), columns=test_scaled.columns.values).set_index([test_scaled.index.values])
    return train_unscaled, validate_unscaled, test_unscaled

def uniform_scaler(train, validate, test):
    '''
    Accepts three dataframes and applies a non-linear transformer to convert values in each dataframe
    to a standard distribution. This will distort correlations and distances within and across features.. 
    Columns containing object data types are dropped, as strings cannot be directly scaled.

    Parameters (train, validate, test) = three dataframes being scaled
    
    Returns (scaler, train_scaled, validate_scaled, test_scaled)
    '''
    train = train.select_dtypes(exclude=['object'])
    validate = validate.select_dtypes(exclude=['object'])
    test = test.select_dtypes(exclude=['object'])
    scaler = QuantileTransformer(n_quantiles=100, output_distribution='uniform', random_state=123, copy=True).fit(train)
    train_scaled = pd.DataFrame(scaler.transform(train), columns=train.columns.values).set_index([train.index.values])
    validate_scaled = pd.DataFrame(scaler.transform(validate), columns=validate.columns.values).set_index([validate.index.values])
    test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns.values).set_index([test.index.values])
    return scaler, train_scaled, validate_scaled, test_scaled

def gaussian_scaler(train, validate, test):
    '''
    Accepts three dataframes and applies a transformer to convert values in each dataframe
    to a gaussian-like distribution. This function defaults to Yeo-Johnson standard normal distribution. 
    Columns containing object data types are dropped, as strings cannot be directly scaled.

    Parameters (train, validate, test) = three dataframes being scaled
    
    Returns (scaler, train_scaled, validate_scaled, test_scaled)
    '''
    train = train.select_dtypes(exclude=['object'])
    validate = validate.select_dtypes(exclude=['object'])
    test = test.select_dtypes(exclude=['object'])
    scaler = PowerTransformer(method='yeo-johnson', standardize=False, copy=True).fit(train)
    train_scaled = pd.DataFrame(scaler.transform(train), columns=train.columns.values).set_index([train.index.values])
    validate_scaled = pd.DataFrame(scaler.transform(validate), columns=validate.columns.values).set_index([validate.index.values])
    test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns.values).set_index([test.index.values])
    return scaler, train_scaled, validate_scaled, test_scaled

def min_max_scaler(train, validate, test):
    '''
    Accepts three dataframes and applies a linear transformer to convert values in each dataframe
    to a value from 0 to 1 while mantaining relative distance between values. 
    Columns containing object data types are dropped, as strings cannot be directly scaled.

    Parameters (train, validate, test) = three dataframes being scaled
    
    Returns (scaler, train_scaled, validate_scaled, test_scaled)
    '''
    train = train.select_dtypes(exclude=['object'])
    validate = validate.select_dtypes(exclude=['object'])
    test = test.select_dtypes(exclude=['object'])    
    scaler = MinMaxScaler(copy=True, feature_range=(0,1)).fit(train)
    train_scaled = pd.DataFrame(scaler.transform(train), columns=train.columns.values).set_index([train.index.values])
    validate_scaled = pd.DataFrame(scaler.transform(validate), columns=validate.columns.values).set_index([validate.index.values])
    test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns.values).set_index([test.index.values])
    return scaler, train_scaled, validate_scaled, test_scaled 

def iqr_robust_scaler(train, validate, test):
    '''
    Accepts three dataframes and applies a linear transformer to convert values in each dataframe
    to a value from 0 to 1 while mantaining relative distance between values. 
    Columns containing object data types are dropped, as strings cannot be directly scaled.

    Parameters (train, validate, test) = three dataframes being scaled
    
    Returns (scaler, train_scaled, validate_scaled, test_scaled)
    '''
    train = train.select_dtypes(exclude=['object'])
    validate = validate.select_dtypes(exclude=['object'])
    test = test.select_dtypes(exclude=['object'])    
    scaler = RobustScaler(quantile_range=(25.0,75.0), copy=True, with_centering=True, with_scaling=True).fit(train)
    train_scaled = pd.DataFrame(scaler.transform(train), columns=train.columns.values).set_index([train.index.values])
    validate_scaled = pd.DataFrame(scaler.transform(validate), columns=validate.columns.values).set_index([validate.index.values])
    test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns.values).set_index([test.index.values])
    return scaler, train_scaled, validate_scaled, test_scaled 

def quantile_scaler_normal(train, validate, test):
    '''
    Accepts three dataframes and applies QuantileTransform() to convert values in each dataframe
    to a normal distribution. 
    Columns containing object data types are dropped, as strings cannot be directly scaled.

    Parameters (train, validate, test) = three dataframes being scaled
    
    Returns (scaler, train_scaled, validate_scaled, test_scaled)
    '''
    # Remove columns with object data types from each dataframe
    train = train.select_dtypes(exclude=['object'])
    validate = validate.select_dtypes(exclude=['object'])
    test = test.select_dtypes(exclude=['object'])
    # Fit the scaler to the train dataframe
    scaler = QuantileTransformer(output_distribution='normal').fit(train)
    # Transform the scaler onto the train, validate, and test dataframes
    train_scaled = pd.DataFrame(scaler.transform(train), columns=train.columns.values).set_index([train.index.values])
    validate_scaled = pd.DataFrame(scaler.transform(validate), columns=validate.columns.values).set_index([validate.index.values])
    test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns.values).set_index([test.index.values])
    return scaler, train_scaled, validate_scaled, test_scaled

def quantile_scaler(train, validate, test):
    '''
    Accepts three dataframes and applies QuantileTransform() to convert values in each dataframe
    to a uniform distribution. 
    Columns containing object data types are dropped, as strings cannot be directly scaled.

    Parameters (train, validate, test) = three dataframes being scaled
    
    Returns (scaler, train_scaled, validate_scaled, test_scaled)
    '''  
    # Remove columns with object data types from each dataframe
    train = train.select_dtypes(exclude=['object'])
    validate = validate.select_dtypes(exclude=['object'])
    test = test.select_dtypes(exclude=['object'])
    # Fit the scaler to the train dataframe
    scaler = QuantileTransformer().fit(train)
    # Transform the scaler onto the train, validate, and test dataframes
    train_scaled = pd.DataFrame(scaler.transform(train), columns=train.columns.values).set_index([train.index.values])
    validate_scaled = pd.DataFrame(scaler.transform(validate), columns=validate.columns.values).set_index([validate.index.values])
    test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns.values).set_index([test.index.values])
    return scaler, train_scaled, validate_scaled, test_scaled