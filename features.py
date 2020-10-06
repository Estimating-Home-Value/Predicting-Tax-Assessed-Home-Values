import pandas as pd 
import numpy as np
from sklearn.feature_selection import SelectKBest, RFE, f_regression 
from sklearn.linear_model import LinearRegression

def select_kbest(X_train_scaled, y_train, k):
    '''
    Takes in the predictors (X_train_scaled), the target (y_train), 
    and the number of features to select (k) 
    and returns the names of the top k selected features based on the SelectKBest class
    '''
    f_selector = SelectKBest(f_regression, k)
    f_selector = f_selector.fit(X_train_scaled, y_train)
    X_train_reduced = f_selector.transform(X_train_scaled)
    f_support = f_selector.get_support()
    f_feature = X_train_scaled.iloc[:,f_support].columns.tolist()
    return f_feature

def rfe(X_train_scaled, y_train, k):
    '''
    Takes in the predictor (X_train_scaled), the target (y_train), 
    and the number of features to select (k).
    Returns the top k features based on the RFE class.
    '''
    lm = LinearRegression()
    rfe = RFE(lm, k)
    # Transforming data using RFE
    X_rfe = rfe.fit_transform(X_train_scaled, y_train)
    #Fitting the data to model
    lm.fit(X_rfe,y_train)
    mask = rfe.support_
    rfe_features = X_train_scaled.loc[:,mask].columns.tolist()
    return rfe_features