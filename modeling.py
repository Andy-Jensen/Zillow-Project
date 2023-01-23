#imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from acquire import wrangle_zillow
from prepare import prep_zillow
from prepare import remove_outliers
from prepare import tts_con

import sklearn.preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression


#Function to prep the data for modeling
def prep_for_modeling():
    '''
    acquire and prep data with functions from explore and prepare
    '''
    zillow=wrangle_zillow()
    zillow=prep_zillow(zillow)
    '''
    remove outliers and get dummies for bedroom count and bathroom count
    '''
    zillow, fences=remove_outliers(zillow)
    zillow = pd.get_dummies(zillow, columns=['bed_count', 'bath_count'])
    '''
    split the data into train, validate, and test
    '''
    m_train, m_val, m_test=tts_con(zillow)
    '''
    split the data into x and y train, validate, and test
    '''
    X_train=m_train.drop(columns=['property_value'])
    y_train=pd.DataFrame(m_train['property_value'])

    X_val=m_val.drop(columns=['property_value'])
    y_val=pd.DataFrame(m_val['property_value'])

    X_test=m_test.drop(columns=['property_value'])
    y_test=pd.DataFrame(m_test['property_value'])
    '''
    scale the data (finished square feet)
    '''
    ss=sklearn.preprocessing.StandardScaler()
    num_cols=['finished_sq_ft']
    ss.fit(X_train[num_cols])

    X_train[num_cols]= ss.transform(X_train[num_cols])
    X_val[num_cols]= ss.transform(X_val[num_cols])
    X_test[num_cols]= ss.transform(X_test[num_cols])
    return X_train, y_train, X_val, y_val, X_test, y_test

#Function for baseline
def get_baseline(t, v):
    '''
    setting a mean and median baseline value for train and validate sets
    '''
    t['base_med']=t['property_value'].median()
    t['Baseline Mean']=t['property_value'].mean()

    v['base_med']=v['property_value'].median()
    v['Baseline Mean']=v['property_value'].mean()
    '''
    calculating RMSE for train and validate baselines
    '''
    med=mean_squared_error(t['property_value'], t['base_med'], squared=False)
    mea=mean_squared_error(t['property_value'], t['Baseline Mean'], squared=False)

    med_v=mean_squared_error(v['property_value'], v['base_med'], squared=False)
    mea_v=mean_squared_error(v['property_value'], v['Baseline Mean'], squared=False)
    '''
    Printing results
    '''
    print(f'Train RMSE for the median baseline is {med}')
    print(f'Train RMSE for the mean baseline is {mea}')
    print('\n')
    print(f'Validate RMSE for the median baseline is {med_v}')
    print(f'Validate RMSE for the mean baseline is {mea_v}')
    '''
    dropping the higher RMSE
    '''
    t=t.drop(columns=['base_med'])
    v=v.drop(columns=['base_med'])
    return t, v

#function for creating models and predictions
def models(xt,yt,xv,yv):
    '''
    create linear regression model and make predictions
    '''
    lm=LinearRegression()
    lm.fit(xt, yt['property_value'])
    yt['Linear Regression']=lm.predict(xt)
    yv['Linear Regression']=lm.predict(xv)
    '''
    create lasso lars model and make predictions
    '''
    ll=LassoLars(fit_intercept=False)
    ll.fit(xt, yt['property_value'])
    yt['Lasso Lars']=ll.predict(xt)
    yv['Lasso Lars']=ll.predict(xv)
    '''
    create tweedie regressor model and make predictions
    '''
    tweed=TweedieRegressor()
    tweed.fit(xt, yt['property_value'])
    yt['Tweedie Regressor']=tweed.predict(xt)
    yv['Tweedie Regressor']=tweed.predict(xv)
    '''
    create additional features for polynomial regression
    '''
    pf=PolynomialFeatures(degree=2)

    X_train_degree2 = pf.fit_transform(xt)
    X_validate_degree2 = pf.transform(xv)
    X_test_degree2 = pf.transform(xt)
    '''
    create the polynomial regression model and make predictions
    '''
    lm2= LinearRegression(normalize=True)
    lm2.fit(X_train_degree2, yt['property_value'])
    yt['Polynomial Regression']=lm2.predict(X_train_degree2)
    yv['Polynomial Regression']=lm2.predict(X_validate_degree2)
    return yt, yv

#function for calculating rmse and the difference
def model_rmse(yt,yv):
    '''
    setting prediction columns and empty lists to append to
    '''
    preds=['Baseline Mean', 'Linear Regression', 'Lasso Lars', 'Tweedie Regressor', 'Polynomial Regression']
    a=[]
    b=[]
    e=[]
    c=[a,b,e]
    '''
    for loop for calculating rmse in y_train and y_validate sets
    '''
    for col in yt[preds]:
        rmse= mean_squared_error(yt['property_value'], yt[col], squared=False)
        rmsev=mean_squared_error(yv['property_value'], yv[col], squared=False)
        b.append(rmse)
        a.append(col)
        e.append(rmsev)
        '''
        assigning all the information to a dataframe for legibility and sorting by the difference
        '''
        rmsedf= pd.DataFrame(data=c, index=['Model', 'RMSE Train', 'RMSE Validate']).T
        rmsedf['Train/Validate Difference']=rmsedf['RMSE Validate']-rmsedf['RMSE Train']
        rmsedf=rmsedf.sort_values(by='Train/Validate Difference')
    return rmsedf

#Function for rmse visualization
def rmse_viz(df):
    '''
    barplot to compare rmse difference
    '''
    sns.barplot(x='Model', y='Train/Validate Difference', data=df)
    plt.title('RMSE difference for Models')
    plt.xticks(rotation=30, ha='right')
    plt.ylabel('Difference')
    plt.show
    
#function for test set
def model_test(xt,yt,xtest,ytest,df):
    '''
    Create and fit model
    '''
    lm=LinearRegression()
    lm.fit(xt, yt['property_value'])
    '''
    get test predictions
    '''
    ytest['lm_preds']=lm.predict(xtest)
    '''
    calculate RMSE for the test set and assign it to a dataframe
    '''
    test_rmse=mean_squared_error(ytest['property_value'], ytest['lm_preds'], squared=False)
    final_model= pd.DataFrame(df.iloc[0,:]).T
    final_model['RMSE Test']= test_rmse
    return final_model