import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

from config import TSOptimizationConfig
from src.preprocessing import get_experiment_data
from src.clusterization import get_cluster_mapping
from src.baseline_model import get_basilne_metrics
from src.global_model import get_global_metrics
from src.utils import baseline_data_split
from src.transformations import transform_dfs

def run_experiment():
    config = TSOptimizationConfig()
    exp_data, exp_data_wide = get_experiment_data()
    clusters_map = get_cluster_mapping(exp_data_wide)

    res_vals = {}
    transform_methods = [None, "diff", "boxcox", "log1p"]
    exp_data_clust = exp_data.assign(clust=lambda df: df['id'].map(clusters_map)).copy()
    for clust, df in tqdm(exp_data_clust.groupby('clust')):
        df_sample = df.drop(columns=["clust"])
        for transform in tqdm(transform_methods):
            train_df, val_df, test_df = baseline_data_split(df_sample, config)
            train_df, val_df, test_df, params = transform_dfs(train_df, val_df, test_df, method=transform)
            baseline_res = get_basilne_metrics(train_df, val_df, test_df, config, params=params)
            res_vals[f"{transform}_clust_{clust}"] = baseline_res.iloc[1:].mean()

    print(res_vals)

    res_vals_global = {}
    transform_methods = [None, "diff", "boxcox", "log1p"]
    exp_data_clust = exp_data.assign(clust=lambda df: df['id'].map(clusters_map)).copy()
    for clust, df in tqdm(exp_data_clust.groupby('clust')):
        df_sample = df.drop(columns=["clust"])
        for transform in tqdm(transform_methods):
            train_df, val_df, test_df = baseline_data_split(df_sample, config)
            train_df, val_df, test_df, params = transform_dfs(train_df, val_df, test_df, method=transform)
            baseline_res = get_global_metrics(train_df, val_df, test_df, config, params=params)
            res_vals_global[f"{transform}_clust_{clust}"] = baseline_res.iloc[1:].mean()

    print(res_vals_global)

if __name__ == "__main__":
    run_experiment()