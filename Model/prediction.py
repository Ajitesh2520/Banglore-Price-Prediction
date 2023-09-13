import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
# import pickle
matplotlib.rcParams["figure.figsize"]=(20,10)
from sklearn.model_selection import train_test_split

df1=pd.read_csv('bengaluru_house_prices.csv')

print(df1.groupby('area_type')['area_type'].agg('count'))

# we r dropping th columns assuming that they play no significant role in deciding the price of the property
df2=df1.drop(['area_type','society','balcony','availability'],axis='columns')
print(df2.head(5))

                       # Data Cleaning Process
# 1) getting rid of na values
print(df2.isnull().sum())
df3=df2.dropna()
print(df3.isnull().sum())

# (+) getting rid of size of flat column
df3['bhk']=df3['size'].apply(lambda x :int(x.split(' ')[0]))
print(df3.head(5))

print(df3['bhk'].unique())

print(df3.total_sqft.unique())

def isFloat(x):
    try:
        float(x)
    except:
        return False
    return True

print(df3[~df3['total_sqft'].apply(isFloat)].head(5))

def convert(x):
    tokens=x.split('-')
    if len(tokens)==2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None

df4=df3.copy()
df4['total_sqft']=df4['total_sqft'].apply(convert)
print(df4.head(10))

df5=df4.copy()
df5['price_per_sqft']=df5['price']/df5['total_sqft']*100000
print(df5.head())


location_stats=df5.groupby('location')['location'].agg('count').sort_values(ascending=False)
print(location_stats)

location_stats_less_than_10=location_stats[location_stats<=10]
df5.location=df5.location.apply(lambda x:'other' if x in location_stats_less_than_10 else x)

# Outlier Removal 

df6=df5[~((df5.total_sqft)/(df5.bhk)<300)]
print(df6.price_per_sqft.describe())

def remove_pps_outliers(df):
    df_out=pd.DataFrame()
    for key,subdf in df.groupby('location'):
        m=np.mean(subdf.price_per_sqft)
        st=np.std(subdf.price_per_sqft)
        reduced_df=subdf[((subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st)))]
        df_out=pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out

df7=remove_pps_outliers(df6)
print(df7.shape)

def remove_bhk_outliers(df):
    exclude_indices=np.array([])
    for location,location_df in df.groupby('location'):
        bhk_stats={}
        for bhk,bhk_df in location_df.groupby('bhk'):
             bhk_stats[bhk]={
                 'mean': np.mean(bhk_df.price_per_sqft),
                 'std': np.std(bhk_df.price_per_sqft),
                 'count':bhk_df.shape[0]
        }
    for bhk,bhk_df in location_df.groupby('bhk'):
        stats=bhk_stats.get(bhk-1)
        if stats and stats['count']>5:
            exclude_indices=np.append(exclude_indices,bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')

df8= remove_bhk_outliers(df7)
print(df8)

# removing bathroom outliers
df8[df8.bath>10]
df9=df8[df8.bath<df8.bhk+2]
print(df9.shape)

df10=df9.drop(['size','price_per_sqft'],axis='columns')
print(df10.head(3))

# Training the model
# 1) managing the location variable using one hot coding
dummies=pd.get_dummies(df10.location)
print(dummies.head())

df11=pd.concat([df10,dummies.drop('other',axis='columns')],axis='columns')
print(df11.head(3))

df12=df11.drop('location',axis='columns')
print(df12.head(3))
print(df12.shape)

x=df12.drop('price',axis='columns')
y=df12.price
print(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=10)
from sklearn.linear_model import LinearRegression
lr_clf=LinearRegression()
lr_clf.fit(x_train,y_train)
print(lr_clf.score(x_test,y_test))

from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

cross_val_score(LinearRegression(), x, y, cv=cv)

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

def find_best_model_using_gridsearchcv(x,y):
    algos = {
        'linear_regression' : {
            'model': LinearRegression(),
            'params': {
                'normalize': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1,2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion' : ['mse','friedman_mse'],
                'splitter': ['best','random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs =  GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(x,y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores,columns=['model','best_score','best_params'])

print(find_best_model_using_gridsearchcv(x,y))

def predict_price(location,sqft,bath,bhk):    
    loc_index = np.where(x.columns==location)[0][0]

    z = np.zeros(len(x.columns))
    z[0] = sqft
    z[1] = bath
    z[2] = bhk
    if loc_index >= 0:
        z[loc_index] = 1

    return lr_clf.predict([z])[0]

print(predict_price('1st Phase JP Nagar',1000, 2, 2))

print(predict_price('Indira Nagar',1000, 3, 3) )


import pickle
with open('banglore_home_prices_model.pickle','wb') as f:
    pickle.dump(lr_clf,f)

import json
columns = {
    'data_columns' : [col.lower() for col in x.columns]
}
with open("columns.json","w") as f:
    f.write(json.dumps(columns))

