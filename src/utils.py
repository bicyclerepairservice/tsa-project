from datetime import datetime
from distutils.util import strtobool
from typing import Sequence, Tuple, Union

import sys
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import TSOptimizationConfig


def convert_tsf_to_dataframe(
    full_file_path_and_name,
    replace_missing_vals_with="NaN",
    value_column_name="series_value",
):
    col_names = []
    col_types = []
    all_data = {}
    line_count = 0
    frequency = None
    found_data_tag = False
    found_data_section = False
    started_reading_data_section = False

    with open(full_file_path_and_name, "r", encoding="cp1252") as file:
        for line in file:
            line = line.strip()

            if line:
                if line.startswith("@"):
                    if not line.startswith("@data"):
                        line_content = line.split(" ")
                        if line.startswith("@attribute"):
                            if (
                                len(line_content) != 3
                            ):
                                raise Exception("Invalid meta-data specification.")

                            col_names.append(line_content[1])
                            col_types.append(line_content[2])
                        else:
                            if (
                                len(line_content) != 2
                            ):
                                raise Exception("Invalid meta-data specification.")

                            if line.startswith("@frequency"):
                                frequency = line_content[1]

                    else:
                        if len(col_names) == 0:
                            raise Exception(
                                "Missing attribute section. Attribute section must come before data."
                            )

                        found_data_tag = True
                elif not line.startswith("#"):
                    if len(col_names) == 0:
                        raise Exception(
                            "Missing attribute section. Attribute section must come before data."
                        )
                    elif not found_data_tag:
                        raise Exception("Missing @data tag.")
                    else:
                        if not started_reading_data_section:
                            started_reading_data_section = True
                            found_data_section = True
                            all_series = []

                            for col in col_names:
                                all_data[col] = []

                        full_info = line.split(":")

                        if len(full_info) != (len(col_names) + 1):
                            raise Exception("Missing attributes/values in series.")

                        series = full_info[len(full_info) - 1]
                        series = series.split(",")

                        if len(series) == 0:
                            raise Exception(
                                "A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series. Missing values should be indicated with ? symbol"
                            )

                        numeric_series = []

                        for val in series:
                            if val == "?":
                                numeric_series.append(replace_missing_vals_with)
                            else:
                                numeric_series.append(float(val))

                        if numeric_series.count(replace_missing_vals_with) == len(
                            numeric_series
                        ):
                            raise Exception(
                                "All series values are missing. A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series."
                            )

                        all_series.append(pd.Series(numeric_series).array)

                        for i in range(len(col_names)):
                            att_val = None
                            if col_types[i] == "numeric":
                                att_val = int(full_info[i])
                            elif col_types[i] == "string":
                                att_val = str(full_info[i])
                            elif col_types[i] == "date":
                                att_val = datetime.strptime(
                                    full_info[i], "%Y-%m-%d %H-%M-%S"
                                )
                            else:
                                raise Exception(
                                    "Invalid attribute type."
                                )
                            if att_val is None:
                                raise Exception("Invalid attribute value.")
                            else:
                                all_data[col_names[i]].append(att_val)

                line_count = line_count + 1

        if line_count == 0:
            raise Exception("Empty file.")
        if len(col_names) == 0:
            raise Exception("Missing attribute section.")
        if not found_data_section:
            raise Exception("Missing series information under data section.")

        all_data[value_column_name] = all_series
        return (
            pd.DataFrame(all_data),
            frequency,
        )


def features_targets__train_idx(
    id_column: pd.Series,
    series_length: int,
    model_horizon: int,
    history_size: int,
) -> np.ndarray:
    """Создание индексов для формирования обучающей выборки (признаков и таргетов) для многомерных временных рядов.
    Args:
        id_column: Колонка с идентификаторами рядов.
        series_length: Общая длина всех рядов.
        model_horizon: Горизонт прогнозирования модели.
        history_size: Размер окна истории.

    Returns:
        Индексы для формирования признаков и таргетов.

    """
    series_start_indices = np.append(
        np.unique(id_column.values, return_index=True)[1], series_length
    )

    features_indices = []
    targets_indices = []
    for i in range(len(series_start_indices) - 1):
        series_start = series_start_indices[i]
        series_end = series_start_indices[i + 1]

        if series_end - series_start < history_size + model_horizon:
            continue  # Пропускаем ряды, которые слишком короткие для формирования окна истории + таргета

        sliding_window = np.lib.stride_tricks.sliding_window_view(
            np.arange(series_start, series_end),
            history_size + model_horizon,
        )

        features_indices.append(sliding_window[:, :history_size])
        targets_indices.append(sliding_window[:, history_size:])

    features_indices = np.vstack(features_indices)
    targets_indices = np.vstack(targets_indices)

    return features_indices, targets_indices


def features__test_idx(
    id_column: pd.Series,
    series_length: int,
    model_horizon: int,
    history_size: int,
) -> np.ndarray:
    """Создание индексов для формирования тестовой выборки для многомерных временных рядов.
    Args:
        id_column: Колонка с идентификаторами рядов.
        series_length: Общая длина всех рядов.
        model_horizon: Горизонт прогнозирования модели.
        history_size: Размер окна истории.

    Returns:
        Индексы для формирования признаков.
        Обратите внимание, что для признаков тестовой выборки нужны только последние history индексов.

    """
    series_start_indices = np.append(
        np.unique(id_column.values, return_index=True)[1], series_length
    )

    features_indices = []
    targets_indices = []
    for i in range(len(series_start_indices) - 1):
        series_start = series_start_indices[i]
        series_end = series_start + history_size + model_horizon

        if series_end - series_start < history_size + model_horizon:
            AssertionError(f"Ряд {i} слишком короткий для формирования окна истории + таргета")

        sliding_window = np.lib.stride_tricks.sliding_window_view(
            np.arange(series_start, series_end),
            history_size + model_horizon,
        )

        features_indices.append(sliding_window[:, :history_size])
        targets_indices.append(sliding_window[:, history_size:])

    features_indices = np.vstack(features_indices)
    targets_indices = np.vstack(targets_indices)

    return features_indices, targets_indices


def get_slice(data: pd.DataFrame, k: Tuple[np.ndarray]) -> np.ndarray:
    """Получение среза из DataFrame по индексам строк и колонок.
    
    Args:
        data: Исходный DataFrame.
        k: Кортеж из двух элементов:
            - Массив индексов строк.
            - Массив индексов колонок или None (если нужны все колонки).
            
    Returns:
        Массив значений из DataFrame по заданным индексам.

    """    
    rows, cols = k
    if cols is None:
        new_data = data.values[rows]
    else:
        new_data = data.iloc[:, cols].values[rows]

    if new_data.ndim == 3:
        new_data = new_data.reshape(new_data.shape[0], -1)

    return new_data

def get_cols_idx(data: pd.DataFrame, columns: Union[str, Sequence[str]]) -> Union[int, np.ndarray]:
    """Получение индексов колонок по их названиям.

    Args:
        data: DataFrame с колонками.
        columns: Название колонки или список названий колонок.

    Returns:
        Индекс колонки или массив индексов колонок.

    """
    if type(columns) is str:
        idx = data.columns.get_loc(columns)
    else:
        idx = data.columns.get_indexer(columns)
    return idx


def get_cols_idx(data: pd.DataFrame, columns: Union[str, Sequence[str]]) -> Union[int, np.ndarray]:
    """Получение индексов колонок по их названиям.

    Args:
        data: DataFrame с колонками.
        columns: Название колонки или список названий колонок.

    Returns:
        Индекс колонки или массив индексов колонок.

    """
    if type(columns) is str:
        idx = data.columns.get_loc(columns)
    else:
        idx = data.columns.get_indexer(columns)
    return idx


def baseline_data_split(
    df: pd.DataFrame, config: TSOptimizationConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df["month"] = df["timestamp"].dt.month
    df["quarter"] = df["timestamp"].dt.quarter
    df["year"] = df["timestamp"].dt.year
    last_year = df["timestamp"].max() - pd.DateOffset(months=config.horizon)
    
    train_val_df = df[df["timestamp"] <= last_year].copy()
    train_df = train_val_df[
        train_val_df["timestamp"] <= last_year - pd.DateOffset(months=config.horizon)
    ].copy()
    val_df = train_val_df[train_val_df["timestamp"] > last_year - pd.DateOffset(months=config.horizon)].copy()
    test_df = df[df["timestamp"] > last_year].copy()

    return train_df, val_df, test_df


def get_features_df_and_targets(
    df: pd.DataFrame,
    features_ids,
    targets_ids,
    id_column: Union[str, Sequence[str]] = "id",
    date_column: Union[str, Sequence[str]] = ["month", "quarter", "year"],
    target_column: str = "target",
):
    # Признаки идентификатора ряда и времени возьмем из таргетных индексов
    features_df_id = get_slice(df, (targets_ids, get_cols_idx(df, id_column)))
    features_time = get_slice(
        df, (targets_ids, get_cols_idx(df, date_column))
    )

    # Лаговые признаки возьмем из индексов признаков. Все доступные из истории
    features_lags = get_slice(
        df,
        (features_ids, get_cols_idx(df, target_column)),
    )

    # Объединим все признаки в один массив
    features = np.hstack([features_df_id, features_time, features_lags])
    categorical_features_idx = np.arange(
        features_df_id.shape[1] + features_time.shape[1]
    )  # Отметим категориальные признаки для CatBoost

    # Сформируем таргеты
    targets = get_slice(df, (targets_ids, get_cols_idx(df, target_column)))
    return features, targets, categorical_features_idx


def get_forecast_df(
    train_val_df: pd.DataFrame, test_df: pd.DataFrame, config: TSOptimizationConfig
) -> pd.DataFrame:
    test_df_to_predict = train_val_df[
        train_val_df["timestamp"] >= test_df["timestamp"].min() - pd.DateOffset(months=config.history)
    ]
    
    new_segments = []
    for current_id in test_df_to_predict["id"].unique():
        segment_df = test_df_to_predict[test_df_to_predict["id"] == current_id].copy()
        last_timestamp = segment_df["timestamp"].max()
    
        new_timestamps = pd.date_range(
            start=last_timestamp + pd.DateOffset(months=1),
            periods=config.horizon,
            freq="ME",
        )
        
        new_segment = pd.DataFrame(
            {
                "id": current_id,
                "timestamp": new_timestamps,
                "target": np.nan,
                "month": new_timestamps.month,
                "quarter": new_timestamps.quarter,
                "year": new_timestamps.year,
            }
        )
        new_segment = pd.concat([segment_df, new_segment], ignore_index=True)
        new_segments.append(new_segment)
    
    return pd.concat(new_segments, ignore_index=True)