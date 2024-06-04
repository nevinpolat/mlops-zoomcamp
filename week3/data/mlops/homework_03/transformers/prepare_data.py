import pandas as pd
import datetime


if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer

@transformer

def clean_dataframe(df: pd.DataFrame):


    # Convert pickup and dropoff datetime columns to datetime type
    df.tpep_dropoff_datetime = pd.to_datetime(df.tpep_dropoff_datetime)
    df.tpep_pickup_datetime = pd.to_datetime(df.tpep_pickup_datetime)

    # Calculate the trip duration in minutes
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    # Filter out trips that are less than 1 minute or more than 60 minutes

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    return df