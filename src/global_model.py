import sys
import numpy as np
import pandas as pd
import catboost as cb
from pathlib import Path
import matplotlib.pyplot as plt
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import TSOptimizationConfig
from src.utils import get_forecast_df
from src.metrics import get_metrics, plot_results
from src.utils import features_targets__train_idx, get_features_df_and_targets, features__test_idx
from src.transformations import inverse_transform_df


def train_cb_model(
    categorical_features_idx: np.ndarray,
    train_features: np.ndarray,
    train_targets: np.ndarray,
    val_features: np.ndarray,
    val_targets: np.ndarray,
) -> cb.CatBoostRegressor:
    cb_model = cb.CatBoostRegressor(
        loss_function="MultiRMSE",
        random_seed=1337,
        verbose=100,
        early_stopping_rounds=50,
        iterations=100,
        cat_features=categorical_features_idx,
    )
    
    train_dataset = cb.Pool(
        data=train_features, label=train_targets, cat_features=categorical_features_idx
    )
    eval_dataset = cb.Pool(data=val_features, label=val_targets, cat_features=categorical_features_idx)
    
    cb_model.fit(
        train_dataset,
        eval_set=eval_dataset,
        use_best_model=True,
        plot=False,
    )
    return cb_model

def get_global_metrics(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: TSOptimizationConfig,
    plot: bool = False,
    params: dict | None = None,
) -> pd.DataFrame:
    train_features_idx, train_targets_idx = features_targets__train_idx(
        id_column=train_df["id"],
        series_length=len(train_df),
        model_horizon=config.step_size,
        history_size=config.history,
    )
    # К валидационной выборке нужно присоединить историю из обучающей выборки
    train_history_df = train_df[
        train_df["timestamp"] >= val_df["timestamp"].min() - pd.DateOffset(months=config.history)
    ]
    history_val_df = pd.concat([train_history_df, val_df], ignore_index=True)
    history_val_df = history_val_df.sort_values(by=["id", "timestamp"])
    val_features_idx, val_targets_idx = features_targets__train_idx(
        id_column=history_val_df["id"],
        series_length=len(history_val_df),
        model_horizon=config.step_size,
        history_size=config.history,
    )
    train_features, train_targets, categorical_features_idx = get_features_df_and_targets(
        train_df,
        train_features_idx,
        train_targets_idx,
    )
    val_features, val_targets, _ = get_features_df_and_targets(
        history_val_df,
        val_features_idx,
        val_targets_idx,
    )

    cb_model = train_cb_model(
        categorical_features_idx,
        train_features,
        train_targets,
        val_features,
        val_targets,
    )
    forecast_df = get_forecast_df(pd.concat([train_df, val_df]), test_df, config)

    for step in range(config.horizon):
        print(f"Шаг прогнозирования {step + 1} из {config.horizon}")
    
        test_features_idx, target_features_idx = features__test_idx(
            id_column=forecast_df["id"],
            series_length=len(forecast_df),
            model_horizon=config.step_size,
            history_size=config.history + step * config.step_size,
        )
        test_features_idx = test_features_idx[:, step:]
    
        test_features, _, _ = get_features_df_and_targets(
            forecast_df,
            test_features_idx,
            target_features_idx,  # Таргеты не нужны
        )
    
        test_preds = cb_model.predict(test_features)
    
        # Заполняем предсказаниями соответствующие места в истории
        forecast_df.iloc[
            target_features_idx.flatten(),  # Берем только последние индексы таргетов
            forecast_df.columns.get_loc("target"),
        ] = test_preds.reshape(-1, 1)


    train_val_df = pd.concat([train_df, val_df])
    if params is not None:
        train_val_df = inverse_transform_df(
            train_val_df,
            params,
            id_column="id",
            time_column="timestamp"
        )
        test_df = inverse_transform_df(
            test_df,
            params,
            id_column="id",
            time_column="timestamp"
        )
        forecast_df = inverse_transform_df(
            forecast_df,
            params,
            id_column="id",
            time_column="timestamp"
        )
    
    if plot:
        plot_results(
            train_df=train_val_df,
            test_df=test_df,
            forecast_df=forecast_df[["id", "timestamp", "target"]].rename(
                columns={"target": "preds"}
            ),
            df_id_column="id",
            df_time_column="timestamp",
            df_target_column="target",
            pred_id_column="id",
            pred_time_column="timestamp",
            num_samples_to_plot=3,
            seed=1337,
        )
    return get_metrics(
        test_df,
        forecast_df=forecast_df[["id", "timestamp", "target"]].rename(
            columns={"target": "preds"}
        ),
        df_id_column="id",
        df_time_column="timestamp",
        df_target_column="target",
        pred_id_column="id",
        pred_time_column="timestamp",
    )
    

