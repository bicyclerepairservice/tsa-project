import random
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def mSMAPE(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    numerator = np.abs(y_true - y_pred)
    denominator = np.maximum(np.abs(y_true) + np.abs(y_pred) + eps, 0.5 + eps) / 2
    msmape = 100 * np.mean(numerator / denominator)

    return msmape


def get_metrics(
    test_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    df_id_column: str = "id",
    df_time_column: str = "timestamp",
    df_target_column: str = "target",
    pred_id_column: str = "unique_id",
    pred_time_column: str = "ds",
) -> dict:
    """Подсчет метрики mSMAPE для всех моделей.
    
    Args:
        test_df: DataFrame с тестовыми данными.
        forecast_df: DataFrame с прогнозами моделей.
        df_id_column: Название колонки с идентификатором ряда в test_df.
        df_time_column: Название колонки с временной меткой в test_df.
        df_target_column: Название колонки с целевой переменной в test_df.
        pred_id_column: Название колонки с идентификатором ряда в forecast_df.
        pred_time_column: Название колонки с временной меткой в forecast_df.

    Returns:
        DataFrame с mSMAPE для всех моделей по всем рядам и по каждому ряду отдельно.

    """
    test_df_with_preds = test_df.merge(
        forecast_df,
        left_on=[df_id_column, df_time_column],
        right_on=[pred_id_column, pred_time_column],
        how="left",
    )

    model_names = [
        col for col in forecast_df.columns if col not in [pred_id_column, pred_time_column]
    ]

    msmape_results = {}

    for model_name in model_names:
        msmape_results[model_name] = {}
        full_pred = test_df_with_preds.dropna()
        msmape_results[model_name]["all"] = mSMAPE(
            full_pred[df_target_column].values,
            full_pred[model_name].values,
        )
        for series_id in test_df[df_id_column].unique():
            test_series = test_df_with_preds[test_df_with_preds[df_id_column] == series_id]
            msmape_series = mSMAPE(
                test_series[df_target_column].values,
                test_series[model_name].values,
            )
            msmape_results[model_name][series_id] = msmape_series

    df_msmape = pd.DataFrame(msmape_results)

    return df_msmape


def plot_results(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    df_id_column: str = "id",
    df_time_column: str = "timestamp",
    df_target_column: str = "target",
    pred_id_column: str = "unique_id",
    pred_time_column: str = "ds",
    num_samples_to_plot: int = 5,
    seed: int = 42,
):
    """Функция для визуализации результатов прогнозирования. 
        Разные модели отображаются разными линиями. 
        Разные ряды — в разных графиках.
    
    Args:
        train_df: DataFrame с тренировочными данными.
        test_df: DataFrame с тестовыми данными.
        forecast_df: DataFrame с прогнозами моделей.
        df_id_column: Название колонки с идентификатором ряда в train_df и test_df.
        df_time_column: Название колонки с временной меткой в train_df и test_df.
        df_target_column: Название колонки с целевой переменной в train_df и test_df.
        pred_id_column: Название колонки с идентификатором ряда в forecast_df.
        pred_time_column: Название колонки с временной меткой в forecast_df.
        num_samples_to_plot: Количество рядов для визуализации.
        seed: Сид для рандома при выборе рядов.
        
    """
    test_df_with_preds = test_df.merge(
        forecast_df,
        left_on=[df_id_column, df_time_column],
        right_on=[pred_id_column, pred_time_column],
        how="left",
    )

    model_names = [
        col for col in forecast_df.columns if col not in [pred_id_column, pred_time_column]
    ]

    random.seed(seed)
    unique_ids = test_df[df_id_column].unique()
    sampled_ids = random.sample(list(unique_ids), num_samples_to_plot)

    for series_id in sampled_ids:
        fig = go.Figure()

        train_series = train_df[train_df[df_id_column] == series_id]
        test_series = test_df_with_preds[test_df_with_preds[df_id_column] == series_id]
        forecast_series = forecast_df[forecast_df[pred_id_column] == series_id]

        for name in model_names:
            fig.add_trace(
                go.Scatter(
                    x=forecast_series[pred_time_column],
                    y=forecast_series[name],
                    mode="lines",
                    line=dict(dash="dash"),
                    name=f"Forecast {name} {series_id}",
                )
            )
        fig.add_trace(
            go.Scatter(
                x=train_series[df_time_column],
                y=train_series[df_target_column],
                mode="lines",
                name=f"Train {series_id}",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=test_series[df_time_column],
                y=test_series[df_target_column],
                mode="lines",
                name=f"Test {series_id}",
            )
        )
        fig.update_layout(
            xaxis_title="Дата",
            yaxis_title="Значение ряда",
            title=f"Результаты прогнозирования для ряда {series_id}",
        )
        fig.show()