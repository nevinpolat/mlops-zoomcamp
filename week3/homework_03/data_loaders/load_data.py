import requests
from io import BytesIO
import pandas as pd

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader


@data_loader
def load_files(**kwargs) -> pd.DataFrame:
    #load data from the github
    response = requests.get(
        f'https://github.com/nevinpolat/mlops-zoomcamp/blob/main/week3/data/yellow_tripdata_2023-03.parquet?raw=true'
            )         

    if response.status_code != 200:
                raise Exception(response.text)

    df = pd.read_parquet(BytesIO(response.content))

    return df