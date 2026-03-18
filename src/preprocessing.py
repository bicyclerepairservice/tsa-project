import sys
import random
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import DATA_PATH
from src.utils import convert_tsf_to_dataframe

def generate_dates(row):
    return pd.to_datetime(row['start_timestamp']) + pd.DateOffset(months=row['month_offset'])

def seed_everything(seed=1337):
    random.seed(seed)
    np.random.seed(seed)

def get_experiment_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    data, frq = convert_tsf_to_dataframe(DATA_PATH)
    start_date = pd.Timestamp('1990-01-01')
    series_to_sample = (
        data
        .explode('series_value')
        .sort_values('start_timestamp')
        .assign(start_timestamp=lambda df: df['start_timestamp'].dt.floor('D'))
        .loc[lambda df: df['start_timestamp'] >= start_date]
        .assign(series_len=lambda df: df.groupby('series_name')['series_value'].transform('count'))
        .loc[lambda df: df['series_len'] == 324, 'series_name']
        .unique().tolist()
    )
    random_samples = random.sample(series_to_sample, 101)
    df_train = (
        data
        .explode('series_value')
        .reset_index(drop=True)
        .loc[lambda df: df['series_name'].isin(random_samples)]
        .assign(
            month_offset=lambda df: df.groupby('series_name').cumcount(),
            date=lambda df: df.apply(generate_dates, axis=1).dt.floor('D'),
            series_value=lambda df: df['series_value'].astype(float),
        )
        .drop(columns=['start_timestamp', 'month_offset'])
        .rename(columns={'date': 'timestamp', 'series_value': 'target', 'series_name': 'id'})
        .sort_values(['id', 'timestamp'])
        .set_index('timestamp')
        .groupby('id')
        .resample('ME')
        .agg({'target': 'mean'})
        .reset_index()
    )
    return df_train, df_train.pivot(index='timestamp', columns='id', values='target').T
