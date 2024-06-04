import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression


if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer

@transformer
def read_dataframe(df: pd.DataFrame):

    #One hot coding
    categorical = ['PULocationID', 'DOLocationID']

    df[categorical] = df[categorical].astype(str)

    
    train_dicts = df[categorical].to_dict(orient='records')
    
    #create feature matrix 
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    
    #Training the data  with linear regression

    target = 'duration'
    y_train = df[target].values

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_train)

    print('y-Intercept is:', lr.intercept_) 

    return dv,lr