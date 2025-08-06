import numpy as np
import pandas as pd
import datetime, os, time
import torch
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import shap
from multiprocessing import Pool, set_start_method, cpu_count

PROCESSES = cpu_count() // 2  # use half of the available CPU cores
BASE_DIR = os.getcwd()
TRAIN_DATA_DIR = os.path.join(BASE_DIR, "kline_data", "train_data")
SUBMISSION_ID_PATH = os.path.join(BASE_DIR, "submission_id.csv")
CACHE_DIR = os.path.join(BASE_DIR, "data_cache")

# 设置 Pandas 未来行为
pd.set_option('future.no_silent_downcasting', True)

# Input data files are available in the read-only "../input/" directory
import os
# for dirname, _, filenames in os.walk(BASE_DIR):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# 动态导入 utils 模块
import sys
sys.path.append(os.path.join(BASE_DIR, 'utils'))
from utils import load_single_symbol_data

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# !pip install xgboost --upgrade  # 确保安装最新版本，支持 GPU
# !pip install ta torch shap

class OptimizedModel:
    def __init__(self):
        self.train_data_path = TRAIN_DATA_DIR
        self.submission_id_path = SUBMISSION_ID_PATH
        self.start_datetime = datetime.datetime(2021, 3, 1, 0, 0, 0)
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

    def get_all_symbol_list(self):
        try:
            if not os.path.exists(self.train_data_path):
                raise FileNotFoundError(f"Data directory not found: {self.train_data_path}")
            parquet_name_list = os.listdir(self.train_data_path)
            if not parquet_name_list:
                print(f"No Parquet files found in {self.train_data_path}")
                return []
            symbol_list = [parquet_name.split(".")[0] for parquet_name in parquet_name_list]
            print(f"Found {len(symbol_list)} symbols: {symbol_list[:5]}...")  # 打印前5个符号
            return symbol_list
        except Exception as e:
            print(f"Error in get_all_symbol_list: {e}")
            return []

    def get_all_symbol_kline(self):
        t0 = datetime.datetime.now()
        all_symbol_list = self.get_all_symbol_list()
        if not all_symbol_list:
            print("No symbols found, exiting.")
            return ([], [], [], [], [], [], [], [], [], [], [])

        time_index = pd.date_range(start=self.start_datetime, end='2024-12-31', freq='15min')
        print(f"Expected time index length: {len(time_index)}")  # 调试：检查预期长度
        
        # 设置 spawn 启动方法
        try:
            set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # 如果已设置，忽略错误

        with Pool() as pool:
            df_list = pool.map(load_single_symbol_data, [(s, self.train_data_path, time_index) for s in all_symbol_list])

        loaded_symbols = []
        for df, s in zip(df_list, all_symbol_list):
            if not df.empty and 'vwap' in df.columns and not df['vwap'].isna().all():
                loaded_symbols.append(s)
            else:
                print(f"{s} failed: empty or missing 'vwap' or all NaN")
        failed_symbols = [s for s in all_symbol_list if s not in loaded_symbols]
        print(f"Failed symbols: {failed_symbols}")

        if not loaded_symbols:
            print("No valid data loaded, returning empty results")
            return ([], [], [], [], [], [], [], [], [], [], [])

        df_open_price = pd.concat(
            [df['open_price'] for df in df_list if not df.empty and 'open_price' in df.columns],
            axis=1).sort_index(ascending=True)
        print(f"df_open_price index dtype: {df_open_price.index.dtype}, shape: {df_open_price.shape}")
        df_open_price.columns = loaded_symbols
        df_open_price = df_open_price.reindex(columns=all_symbol_list, fill_value=0).reindex(time_index, method='ffill').infer_objects(copy=False)
        time_arr = pd.to_datetime(df_open_price.index).values

        def align_df(dfs, valid_symbols, key):
            valid_dfs = [df[key] for df, s in zip(dfs, all_symbol_list) if not df.empty and key in df.columns and s in valid_symbols]
            if not valid_dfs:
                print(f"No valid data for {key}, filling with zeros")
                return np.zeros((len(time_index), len(all_symbol_list)))
            df = pd.concat(valid_dfs, axis=1).sort_index(ascending=True)
            df.columns = valid_symbols
            return df.reindex(columns=all_symbol_list, fill_value=0).reindex(time_index, method='ffill').infer_objects(copy=False).values

        vwap_arr = align_df(df_list, loaded_symbols, 'vwap')
        amount_arr = align_df(df_list, loaded_symbols, 'amount')
        atr_arr = align_df(df_list, loaded_symbols, 'atr')
        macd_arr = align_df(df_list, loaded_symbols, 'macd')
        buy_volume_arr = align_df(df_list, loaded_symbols, 'buy_volume')
        volume_arr = align_df(df_list, loaded_symbols, 'volume')
        bb_upper_arr = align_df(df_list, loaded_symbols, 'bb_upper')
        bb_lower_arr = align_df(df_list, loaded_symbols, 'bb_lower')
        momentum_1h_arr = align_df(df_list, loaded_symbols, '1h_momentum')

        print(f"Finished get all symbols kline, time elapsed: {datetime.datetime.now() - t0}")
        return all_symbol_list, time_arr, vwap_arr, amount_arr, atr_arr, macd_arr, buy_volume_arr, volume_arr, bb_upper_arr, bb_lower_arr, momentum_1h_arr

    def weighted_spearmanr(self, y_true, y_pred):
        n = len(y_true)
        r_true = pd.Series(y_true).rank(ascending=False, method='average')
        r_pred = pd.Series(y_pred, index=r_true.index).rank(ascending=False, method='average')
        pos = np.arange(n)
        x = 2 * (pos - 1) / (n - 1) - 1
        w = x ** 2
        w_sum = w.sum()
        mu_true = (w * r_true).sum() / w_sum
        mu_pred = (w * r_pred).sum() / w_sum
        cov = (w * (r_true - mu_true) * (r_pred - mu_pred)).sum()
        var_true = (w * (r_true - mu_true) ** 2).sum()
        var_pred = (w * (r_pred - mu_pred) ** 2).sum()
        return cov / np.sqrt(var_true * var_pred) if var_true * var_pred > 0 else 0

    def train(self, df_target, df_4h_momentum, df_7d_momentum, df_amount_sum, df_vol_momentum, df_atr, df_macd,
              df_buy_pressure, df_bb_upper, df_bb_lower, df_1h_momentum):
        common_index = df_target.index.intersection(df_4h_momentum.index).intersection(df_7d_momentum.index).intersection(
            df_amount_sum.index).intersection(df_vol_momentum.index).intersection(df_atr.index).intersection(
            df_macd.index).intersection(df_buy_pressure.index).intersection(df_bb_upper.index).intersection(
            df_bb_lower.index).intersection(df_1h_momentum.index)
        df_target = df_target.loc[common_index]
        df_4h_momentum = df_4h_momentum.loc[common_index]
        df_7d_momentum = df_7d_momentum.loc[common_index]
        df_amount_sum = df_amount_sum.loc[common_index]
        df_vol_momentum = df_vol_momentum.loc[common_index]
        df_atr = df_atr.loc[common_index]
        df_macd = df_macd.loc[common_index]
        df_buy_pressure = df_buy_pressure.loc[common_index]
        df_bb_upper = df_bb_upper.loc[common_index]
        df_bb_lower = df_bb_lower.loc[common_index]
        df_1h_momentum = df_1h_momentum.loc[common_index]

        factor1_long = df_4h_momentum.stack()
        factor2_long = df_7d_momentum.stack()
        factor3_long = df_amount_sum.stack()
        factor4_long = df_vol_momentum.stack()
        factor5_long = df_atr.stack()
        factor6_long = df_macd.stack()
        factor7_long = df_buy_pressure.stack()
        factor8_long = (df_bb_upper - df_bb_lower).stack()  # Bollinger Band 宽度
        factor9_long = df_1h_momentum.stack()

        target_long = df_target.stack()

        factor1_long.name = '4h_momentum'
        factor2_long.name = '7d_momentum'
        factor3_long.name = 'amount_sum'
        factor4_long.name = 'vol_momentum'
        factor5_long.name = 'atr'
        factor6_long.name = 'macd'
        factor7_long.name = 'buy_pressure'
        factor8_long.name = 'bb_width'
        factor9_long.name = '1h_momentum'
        target_long.name = 'target'

        data = pd.concat([factor1_long, factor2_long, factor3_long, factor4_long, factor5_long, factor6_long,
                         factor7_long, factor8_long, factor9_long, target_long], axis=1)
        print(f"Data shape before fillna: {data.shape}")
        data = data.fillna(0).infer_objects(copy=False)  # 修复警告
        print(f"Data shape after fillna: {data.shape}")

        X = data[['4h_momentum', '7d_momentum', 'amount_sum', 'vol_momentum', 'atr', 'macd', 'buy_pressure', 'bb_width', '1h_momentum']]
        y = data['target'].replace([np.inf, -np.inf], 0)

        print(f"X shape: {X.shape}, y shape: {y.shape}")
        X_scaled = self.scaler.fit_transform(X)

        tscv = TimeSeriesSplit(n_splits=10)
        best_score = -np.inf
        best_model = None

        for train_idx, val_idx in tscv.split(X_scaled):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

            y_train_clean = y_train.fillna(0)
            sample_weight = np.where(
                (y_train_clean > y_train_clean.quantile(0.95)) | (y_train_clean < y_train_clean.quantile(0.05)), 3, 1)

            model = xgb.XGBRegressor(
                objective='reg:squarederror',
                learning_rate=0.01,
                max_depth=8,
                subsample=0.8,
                n_estimators=500,
                reg_lambda=1.0,
                tree_method='gpu_hist',  # 启用 GPU 训练
                gpu_id=0,  # 指定 GPU ID
                early_stopping_rounds=20,
                random_state=42
            )
            model.fit(X_train, y_train, sample_weight=sample_weight, eval_set=[(X_val, y_val)], verbose=False)

            y_pred_val = model.predict(X_val)
            score = self.weighted_spearmanr(y_val, y_pred_val)
            if score > best_score:
                best_score = score
                best_model = model

        print(f"Best validation Spearman score: {best_score:.4f}")

        data['y_pred'] = best_model.predict(X_scaled)
        data['y_pred'] = data['y_pred'].replace([np.inf, -np.inf], 0).fillna(0).infer_objects(copy=False)
        data['y_pred'] = data['y_pred'].ewm(span=10).mean()
        print(f"data shape after prediction: {data.shape}")

        df_submit = data.reset_index(level=0)
        print(f"df_submit shape: {df_submit.shape}, sample: {df_submit.head().to_string()}")
        df_submit = df_submit[['level_0', 'y_pred']]
        df_submit['symbol'] = df_submit.index.values
        df_submit = df_submit[['level_0', 'symbol', 'y_pred']]
        df_submit.columns = ['datetime', 'symbol', 'predict_return']
        df_submit = df_submit[df_submit['datetime'] >= self.start_datetime]
        df_submit["id"] = df_submit["datetime"].dt.strftime("%Y%m%d%H%M%S") + "_" + df_submit["symbol"]
        print(f"df_submit shape after filtering: {df_submit.shape}, sample IDs: {df_submit['id'].head().to_string()}")

        if os.path.exists(self.submission_id_path):
            df_submission_id = pd.read_csv(self.submission_id_path)
            id_list = df_submission_id["id"].tolist()
            print(f"Submission ID sample: {df_submission_id['id'].head().to_string()}")
            df_submit_competion = df_submit[df_submit['id'].isin(id_list)]
            missing_elements = list(set(id_list) - set(df_submit_competion['id']))
            print(f"Missing IDs: {len(missing_elements)}")
            new_rows = pd.DataFrame({'id': missing_elements, 'predict_return': [0] * len(missing_elements)})
            df_submit_competion = pd.concat([df_submit_competion, new_rows], ignore_index=True).infer_objects(copy=False)
        else:
            print(f"Warning: {self.submission_id_path} not found. Saving submission without ID filtering.")
            df_submit_competion = df_submit

        print(f"df_submit_competion shape: {df_submit_competion.shape}, content: {df_submit_competion.head().to_string()}")
        output_path = os.path.join(CACHE_DIR, "submit.csv")
        df_submit_competion.to_csv(output_path, index=False)
        print(f"Submission file saved to: {output_path}")

        df_check = data.reset_index(level=0)
        df_check = df_check[['level_0', 'target']]
        df_check['symbol'] = df_check.index.values
        df_check = df_check[['level_0', 'symbol', 'target']]
        df_check.columns = ['datetime', 'symbol', 'true_return']
        df_check = df_check[df_check['datetime'] >= self.start_datetime]
        df_check["id"] = df_check["datetime"].dt.strftime("%Y%m%d%H%M%S") + "_" + df_check["symbol"]
        df_check = df_check[['id', 'true_return']]
        df_check.to_csv(os.path.join(BASE_DIR, "check.csv"), index=False)

        rho_overall = self.weighted_spearmanr(data['target'], data['y_pred'])
        print(f"Weighted Spearman correlation coefficient: {rho_overall:.4f}")

        explainer = shap.Explainer(best_model)
        shap_values = explainer(X_scaled)
        shap.summary_plot(shap_values, X.columns)

    def run(self):
        all_symbol_list, time_arr, vwap_arr, amount_arr, atr_arr, macd_arr, buy_volume_arr, volume_arr, bb_upper_arr, bb_lower_arr, momentum_1h_arr = self.get_all_symbol_kline()
        if not all_symbol_list:
            print("No data loaded, exiting.")
            return

        print(f"all_symbol_list length: {len(all_symbol_list)}, vwap_arr shape: {vwap_arr.shape}")
        df_vwap = pd.DataFrame(vwap_arr, columns=all_symbol_list, index=time_arr)
        df_amount = pd.DataFrame(amount_arr, columns=all_symbol_list, index=time_arr)
        df_atr = pd.DataFrame(atr_arr, columns=all_symbol_list, index=time_arr)
        df_macd = pd.DataFrame(macd_arr, columns=all_symbol_list, index=time_arr)
        df_buy_volume = pd.DataFrame(buy_volume_arr, columns=all_symbol_list, index=time_arr)
        df_volume = pd.DataFrame(volume_arr, columns=all_symbol_list, index=time_arr)
        df_bb_upper = pd.DataFrame(bb_upper_arr, columns=all_symbol_list, index=time_arr)
        df_bb_lower = pd.DataFrame(bb_lower_arr, columns=all_symbol_list, index=time_arr)
        df_1h_momentum = pd.DataFrame(momentum_1h_arr, columns=all_symbol_list, index=time_arr)

        windows_1d = 4 * 24 * 1
        windows_7d = 4 * 24 * 7
        windows_4h = 4 * 4

        df_4h_momentum = (df_vwap / df_vwap.shift(windows_4h) - 1).replace([np.inf, -np.inf], np.nan).fillna(0).infer_objects(copy=False)
        df_7d_momentum = (df_vwap / df_vwap.shift(windows_7d) - 1).replace([np.inf, -np.inf], np.nan).fillna(0).infer_objects(copy=False)
        df_amount_sum = df_amount.rolling(windows_7d).sum().replace([np.inf, -np.inf], np.nan).fillna(0).infer_objects(copy=False)
        df_vol_momentum = (df_amount / df_amount.shift(windows_1d) - 1).replace([np.inf, -np.inf], np.nan).fillna(0).infer_objects(copy=False)
        df_buy_pressure = (df_buy_volume - (df_volume - df_buy_volume)).replace([np.inf, -np.inf], np.nan).fillna(0).infer_objects(copy=False)
        df_24hour_rtn = (df_vwap / df_vwap.shift(windows_1d) - 1).replace([np.inf, -np.inf], np.nan).fillna(0).infer_objects(copy=False)

        self.train(df_24hour_rtn.shift(-windows_1d), df_4h_momentum, df_7d_momentum, df_amount_sum, df_vol_momentum, df_atr,
                   df_macd, df_buy_pressure, df_bb_upper, df_bb_lower, df_1h_momentum)

if __name__ == '__main__':
    # print("Input directory contents:", os.listdir("/kaggle/input/avenir-hku-web/"))
    # print("Train data directory contents:", os.listdir("/kaggle/input/avenir-hku-web/kline_data/train_data"))
    model = OptimizedModel()
    model.run()