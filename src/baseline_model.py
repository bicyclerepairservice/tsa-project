import sys
import pandas as pd
from pathlib import Path
from statsforecast import StatsForecast
from statsforecast.models import AutoETS, AutoTheta, Naive, SeasonalNaive
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import TSOptimizationConfig
from src.metrics import get_metrics, plot_results
from src.transformations import inverse_transform_df

def get_basilne_metrics(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame, 
    config: TSOptimizationConfig,
    plot: bool = False,
    params: dict | None = None,
) -> pd.DataFrame:
    list_models = ["AutoETS", "AutoTheta", "Naive", "SeasonalNaive"]
    train_df_for_statsforecast = pd.concat([train_df, val_df]).rename(
        columns={"timestamp": "ds", "id": "unique_id", "target": "y"}
    )
    sf = StatsForecast(
        models=[
            AutoETS(season_length=config.season_len),
            AutoTheta(season_length=config.season_len),
            Naive(),
            SeasonalNaive(season_length=config.season_len),
        ],
        freq="ME",
        verbose=True,
    )
    sf.fit(train_df_for_statsforecast)
    preds = sf.predict(h=config.horizon)

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

        for col in list_models:
            temp = preds[["ds", "unique_id", col]].rename(columns={col: "target"}).copy()
            temp_transformed = inverse_transform_df(
                temp,
                params,
                id_column="unique_id",
                time_column="ds"
            )
            preds[col] = temp_transformed["target"].values
    
    if plot:
        plot_results(
            train_df=train_val_df,
            test_df=test_df,
            forecast_df=preds,
            df_id_column="id",
            df_time_column="timestamp",
            df_target_column="target",
            pred_id_column="unique_id",
            pred_time_column="ds",
            num_samples_to_plot=3,
            seed=1337,
        )
    return get_metrics(test_df, preds)