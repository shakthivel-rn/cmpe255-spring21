import pandas as pd
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

class CarPrice:

    def __init__(self):
        self.df = pd.read_csv('data/data.csv')
        print(f'${len(self.df)} lines loaded')
        self.base = ['engine_hp', 'engine_cylinders', 'highway_mpg', 'city_mpg', 'popularity']

    def trim(self):
        self.df.columns = self.df.columns.str.lower().str.replace(' ', '_')
        string_columns = list(self.df.dtypes[self.df.dtypes == 'object'].index)
        for col in string_columns:
            self.df[col] = self.df[col].str.lower().str.replace(' ', '_')

    def validate(self):
        np.random.seed(2)

        n = len(self.df)

        n_val = 5
        n_train = n - n_val

        idx = np.arange(n)
        np.random.shuffle(idx)

        df_shuffled = self.df.iloc[idx]

        df_train = df_shuffled.iloc[:n_train].copy()
        df_val = df_shuffled.iloc[n_train:].copy()
   
        y_train = np.log1p(df_train.msrp.values)
        y_val = np.log1p(df_val.msrp.values)
        result_msrp = df_val['msrp']
        
        del df_train['msrp']
        del df_val['msrp']

        return y_train, y_val, df_train, df_val, result_msrp

    def linear_regression(self, X, y):
        ones = np.ones(X.shape[0])
        X = np.column_stack([ones, X])

        XTX = X.T.dot(X)
        XTX_inv = np.linalg.inv(XTX)
        w = XTX_inv.dot(X.T).dot(y)
    
        return w[0], w[1:]

    def prepare_X(self, df):
        df_num = df[self.base]
        df_num = df_num.fillna(0)
        X = df_num.values
        return X

car_price = CarPrice()
car_price.trim()
y_train, y_val, df_train, df_val, result_msrp = car_price.validate()
X_train = car_price.prepare_X(df_train)
w_0, w = car_price.linear_regression(X_train, y_train)
X_val = car_price.prepare_X(df_val)
y_pred = w_0 + X_val.dot(w)
y_pred = np.exp(y_pred) - 1
y_val = np.exp(y_val) - 1
result = df_val
result['msrp'] = result_msrp
result['msrp_pred'] = y_pred
result = result[['engine_cylinders', 'transmission_type', 'driven_wheels', 'number_of_doors', 'market_category', 'vehicle_size', 'vehicle_style', 'highway_mpg', 'city_mpg', 'popularity', 'msrp', 'msrp_pred']]
print(result)





