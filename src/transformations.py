import numpy as np
import pandas as pd
from scipy.stats import boxcox
from scipy.special import inv_boxcox

def transform_dfs(train_df, val_df, test_df, method: str| None = None):
    """
    method: 'log1p', 'boxcox', 'diff'
    """

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    if method is None:
        return train_df, val_df, test_df, None

    params = {}

    def process_group(train_g, val_g, test_g):
        series_id = train_g["id"].iloc[0]

        y_train = train_g["target"].values
        y_val = val_g["target"].values
        y_test = test_g["target"].values

        if method == "log1p":
            train_g["target"] = np.log1p(y_train)
            val_g["target"] = np.log1p(y_val)
            test_g["target"] = np.log1p(y_test)

            params[series_id] = {"method": "log1p"}

        elif method == "boxcox":
            shift = 0
            if np.any(y_train <= 0):
                shift = abs(y_train.min()) + 1
                y_train = y_train + shift
                y_val = y_val + shift
                y_test = y_test + shift

            y_train_bc, lam = boxcox(y_train)

            y_val_bc = boxcox(y_val, lmbda=lam)
            y_test_bc = boxcox(y_test, lmbda=lam)

            train_g["target"] = y_train_bc
            val_g["target"] = y_val_bc
            test_g["target"] = y_test_bc

            params[series_id] = {
                "method": "boxcox",
                "lambda": lam,
                "shift": shift
            }

        elif method == "diff":
            train_diff = np.diff(y_train, prepend=y_train[0])
            val_diff = np.diff(y_val, prepend=y_train[-1])
            test_diff = np.diff(y_test, prepend=y_val[-1])

            train_g["target"] = train_diff
            val_g["target"] = val_diff
            test_g["target"] = test_diff

            params[series_id] = {
                "method": "diff",
                "last_train": y_train[-1],
                "last_val": y_val[-1]
            }

        else:
            raise ValueError("Unknown method")

        return train_g, val_g, test_g

    train_out, val_out, test_out = [], [], []

    for series_id in train_df["id"].unique():
        train_g = train_df[train_df["id"] == series_id]
        val_g = val_df[val_df["id"] == series_id]
        test_g = test_df[test_df["id"] == series_id]

        t, v, te = process_group(train_g, val_g, test_g)

        train_out.append(t)
        val_out.append(v)
        test_out.append(te)

    return (
        pd.concat(train_out),
        pd.concat(val_out),
        pd.concat(test_out),
        params
    )

def inverse_transform_df(df, params, id_column, time_column):
    """
    df: DataFrame с колонками [id, timestamp, target, ...]
        target — это предсказания (в трансформированном пространстве)

    params: словарь из transform_dfs()

    Возвращает df с восстановленным target
    """

    df = df.copy()
    restored_groups = []

    if params is None:
        return df

    for series_id, group in df.groupby(id_column):
        group = group.sort_values(time_column).copy()
        p = params[series_id]

        y_hat = group["target"].values

        if p["method"] == "log1p":
            group["target"] = np.expm1(y_hat)

        elif p["method"] == "boxcox":
            y = inv_boxcox(y_hat, p["lambda"])
            if p["shift"] != 0:
                y -= p["shift"]
            group["target"] = y

        elif p["method"] == "diff":
            # ВАЖНО: кумулятивное восстановление
            restored = []
            last = p["last_train"]  # точка опоры

            for diff_val in y_hat:
                val = last + diff_val
                restored.append(val)
                last = val  # обновляем состояние

            group["target"] = restored

        else:
            raise ValueError("Unknown method")

        restored_groups.append(group)

    return pd.concat(restored_groups).sort_index()
