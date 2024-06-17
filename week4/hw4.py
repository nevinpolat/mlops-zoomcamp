#!/usr/bin/env python
# coding: utf-8


import pickle
import pandas as pd
import os
import sys

year = int(sys.argv[1]) # 2023
month = int(sys.argv[2]) # 3

input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
output_file = f'output/yellow_tripprediction_{year:04d}-{month:02d}.parquet'

#loading pickle file
with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)



#location id and drop off location id are categorical features
categorical = ['PULocationID', 'DOLocationID']

#remove outliers
#fill blank values
def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df



#download yellow march 2023 taxi data
df = read_data(input_file)

df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')


#categorical features turn into dictionary 
dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)

print(f'predicted mean duration for year {year} month {month} is:', y_pred.mean())



df_result = pd.DataFrame()
df_result['ride_id'] = df['ride_id']
df_result['predicted_duration'] = y_pred

os.makedirs('output', exist_ok=True)
df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)








