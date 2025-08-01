import numpy as np
import pandas as pd
import datetime, os, time
import multiprocessing as mp
import torch
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import shap

class OptimizedModel:
    def __init__(self):
        self.train_data_path = "/kaggle/input/avenir-hku-web/kline_data/train_data"
        self.submission_id_path = "/kaggle/input/avenir-hku-web/submission_id.csv"
        self.start_datetime = datetime.datetime(2021, 3, 1, 0, 0, 0)
        self.scaler = StandardScaler()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
    
    def get_all_symbol_list(self):
        try:
            parquet_name_list = os.listdir(self.train_data_path)
            symbol_list = [parquet_name.split(".")[0] for parquet_name in parquet_name_list]
            return symbol_list
        except Exception as e:
            print(f"Error in get_all_symbol_list: {e}")
            return []

    def compute_factors_torch(self, df):
        # 转换为张量 convert to tensor
        close = torch.tensor(df['close_price'].values, dtype=torch.float32, device=self.device)
        volume = torch.tensor(df['volume'].values, dtype=torch.float32, device=self.device)
        amount = torch.tensor(df['amount'].values, dtype=torch.float32, device=self.device)
        high = torch.tensor(df['high_price'].values, dtype=torch.float32, device=self.device)
        low = torch.tensor(df['low_price'].values, dtype=torch.float32, device=self.device)
        buy_volume = torch.tensor(df['buy_volume'].values, dtype=torch.float32, device=self.device)
        
        # VWAP - Volume-Weighted Average Price: https://www.investopedia.com/terms/v/vwap.asp
        ## Average price of the security has traded at throughout the day weighted by volume
        vwap = torch.where(volume > 0, amount / volume, close)
        vwap = torch.where(torch.isfinite(vwap), vwap, close)
        
        # RSI - Relative Strength Index (Full Implementation, 14 day window): https://www.investopedia.com/terms/r/rsi.asp
        ## Momentum indicator to measure how fast & how much a stock moved recently
        ## RSI > 70 -> Asset is overbought, possible trend reversal
        ## RSI < 30 -> Asset maybe oversold, possible reversal outward
        delta = torch.diff(close, prepend=close[:1]) # calculate delta between consecutive elements
        gain = torch.where(delta > 0, delta, torch.tensor(0.0, device=self.device))
        loss = torch.where(delta < 0, -delta, torch.tensor(0.0, device=self.device))
        
        init_gain = gain[:14].mean() if len(gain) > 14 else torch.tensor(0.0, device=self.device)
        init_loss = loss[:14].mean() if len(gain) > 14 else torch.tensor(0.0, device=self.device)
        
        avg_gain = torch.zeros_like(close, device=self.device)
        avg_loss = torch.zeros_like(close, device=self.device)
        avg_gain[:14] = init_gain
        avg_loss[:14] = init_loss
        for i in range(14, len(close)): # Wilder's smoothing method -> less reactive than standard ema, more stable
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
        
        rs = torch.where(avg_loss > 0, avg_gain / avg_loss, torch.tensor(0.0, device=self.device))
        rsi = 100 - 100 / (1 + rs)
        rsi = torch.where(torch.isnan(rsi), torch.tensor(50.0, device=self.device), rsi) # set to neutral if nan
        
        # MACD - Moving Average Convergence Divergence: https://www.investopedia.com/terms/m/macd.asp
        ## Momentum + trend-following indicator widely used to spot trade direction, strength, as well as buy/sell signals
        ema12 = torch.zeros_like(close, device=self.device)
        ema26 = torch.zeros_like(close, device=self.device)
        alpha12 = 2 / (12 + 1)
        alpha26 = 2 / (26 + 1)
        ema12[:12] = close[:12].mean()
        ema26[:26] = close[:26].mean()
        for i in range(12, len(close)):  # short term ema
            ema12[i] = alpha12 * close[i] + (1 - alpha12) * ema12[i-1]
        for i in range(26, len(close)): # long term ema
            ema26[i] = alpha26 * close[i] + (1 - alpha26) * ema26[i-1]
        macd = ema12 - ema26
        macd = torch.where(torch.isnan(macd), torch.tensor(0.0, device=self.device), macd)
        
        # ATR - Average True Range: https://www.investopedia.com/terms/a/atr.asp
        ## Measure of Volatility
        ## Price moves a lot -> High ATR, Price mostly flat -> Low ATR
        tr = torch.max(high - low, torch.max(torch.abs(high - close), torch.abs(low - close)))
        atr = torch.zeros_like(tr, device=self.device)
        for i in range(14, len(tr)): # Wilder's 
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
        atr = torch.where(torch.isnan(atr), torch.tensor(0.0, device=self.device), atr) # set to neutral if nan
        
        # Buy Ratio
        ## Estimate of how much total trading volume is made up of buying pressure vs. selling
        ## Buy Volume: trades that happened at or near ask price -> if relatively high, could signal upward momentum
        ## Sell Volume: trades that happed at or near bid price -> if relatively high, could signal downward momentum
        buy_ratio = torch.where(volume > 0, buy_volume / volume, torch.tensor(0.5, device=self.device))
        
        # VWAP Deviation
        ## Measure of how far current price is from VWAP
        ## Price > VWAP - Buyers pushing up price -> potentially overbought
        ## Price < VWAP - Sellers pushing down price -> potentially oversold
        ## Large deviations can signal mean reversion
        vwap_deviation = (close - vwap) / torch.where(vwap != 0, vwap, torch.tensor(1.0, device=self.device))
        vwap_deviation = torch.where(torch.isfinite(vwap_deviation), vwap_deviation, torch.tensor(0.0, device=self.device))
        
        # Convert torch tensors to dataframes
        df['vwap'] = vwap.cpu().numpy()
        df['rsi'] = rsi.cpu().numpy()
        df['macd'] = macd.cpu().numpy()
        df['atr'] = atr.cpu().numpy()
        df['buy_ratio'] = buy_ratio.cpu().numpy()
        df['vwap_deviation'] = vwap_deviation.cpu().numpy()
        return df

    def get_single_symbol_kline_data(self, symbol):
        try:
            df = pd.read_parquet(f"{self.train_data_path}/{symbol}.parquet")
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            df = df.astype(np.float64)
            required_cols = ['open_price', 'high_price', 'low_price', 'close_price', 'volume', 'amount', 'buy_volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"{symbol} missing columns: {missing_cols}")
                return pd.DataFrame(columns=required_cols + ['vwap', 'rsi', 'macd', 'buy_ratio', 'vwap_deviation', 'atr'])

            # smooth data by removing extreme outliers
            df['close_price'] = df['close_price'].clip(df['close_price'].quantile(0.01), df['close_price'].quantile(0.99))
            df['volume'] = df['volume'].clip(df['volume'].quantile(0.01), df['volume'].quantile(0.99))

            # Calculate the indicators
            df = self.compute_factors_torch(df)
            print(f"Loaded data for {symbol}, shape: {df.shape}, vwap NaNs: {df['vwap'].isna().sum()}")
            return df
        except Exception as e:
            print(f"Error loading {symbol}: {e}")
            return pd.DataFrame(columns=['open_price', 'high_price', 'low_price', 'close_price', 'volume', 'amount', 'buy_volume', 'vwap', 'rsi', 'macd', 'buy_ratio', 'vwap_deviation', 'atr'])

    def get_all_symbol_kline(self):
        t0 = datetime.datetime.now()
        pool = mp.Pool(4)
        all_symbol_list = self.get_all_symbol_list()
        if not all_symbol_list:
            print("No symbols found, exiting.")
            pool.close()
            return [], [], [], [], [], [], [], [], [], []

        # get symbol data + indicators 
        df_list = [pool.apply_async(self.get_single_symbol_kline_data, (symbol,)) for symbol in all_symbol_list]
        pool.close()
        pool.join()

        loaded_symbols = []
        for async_result, symbol in zip(df_list, all_symbol_list):
            df = async_result.get()
            if not df.empty and 'vwap' in df.columns:
                loaded_symbols.append(symbol)
            else:
                print(f"{symbol} failed: empty or missing 'vwap'")
        failed_symbols = [s for s in all_symbol_list if s not in loaded_symbols]
        print(f"Failed symbols: {failed_symbols}")

        # Clean data
        time_index = pd.date_range(start=self.start_datetime, end='2024-12-31', freq='15min') # 15 min time index
        df_results = [async_result.get() for async_result in df_list]
        df_open_price = pd.concat([
            result['open_price'] 
            for result in df_results 
            if not result.empty and 'open_price' in result.columns
        ], axis=1).sort_index(ascending=True) # get open price dfs for each symbol
        print(f"df_open_price index dtype: {df_open_price.index.dtype}, shape: {df_open_price.shape}")
        df_open_price.columns = loaded_symbols
        df_open_price = df_open_price.reindex(columns=all_symbol_list, fill_value=0).reindex(time_index, method='ffill') # unify time steps
        time_arr = pd.to_datetime(df_open_price.index).values

        def align_df(arr, valid_symbols, key):
            valid_dfs = [df[key] for df, s in zip([i.get() for i in df_list], all_symbol_list) if not df.empty and key in df.columns and s in valid_symbols]
            if not valid_dfs:
                print(f"No valid data for {key}, filling with zeros")
                return np.zeros((len(time_index), len(all_symbol_list)))
            df = pd.concat(valid_dfs, axis=1).sort_index(ascending=True)
            df.columns = valid_symbols
            return df.reindex(columns=all_symbol_list, fill_value=0).reindex(time_index, method='ffill').values

        vwap_arr = align_df(df_list, loaded_symbols, 'vwap')
        amount_arr = align_df(df_list, loaded_symbols, 'amount')
        atr_arr = align_df(df_list, loaded_symbols, 'atr')
        macd_arr = align_df(df_list, loaded_symbols, 'macd')
        buy_volume_arr = align_df(df_list, loaded_symbols, 'buy_volume')
        volume_arr = align_df(df_list, loaded_symbols, 'volume')

        print(f"Finished get all symbols kline, time elapsed: {datetime.datetime.now() - t0}")
        return all_symbol_list, time_arr, vwap_arr, amount_arr, atr_arr, macd_arr, buy_volume_arr, volume_arr

    def weighted_spearmanr(self, y_true, y_pred):
        n = len(y_true)
        r_true = pd.Series(y_true).rank(ascending=False, method='average')
        r_pred = pd.Series(y_pred).rank(ascending=False, method='average')
        x = 2 * (r_true - 1) / (n - 1) - 1
        w = x ** 2
        w_sum = w.sum()
        mu_true = (w * r_true).sum() / w_sum
        mu_pred = (w * r_pred).sum() / w_sum
        cov = (w * (r_true - mu_true) * (r_pred - mu_pred)).sum()
        var_true = (w * (r_true - mu_true)**2).sum()
        var_pred = (w * (r_pred - mu_pred)**2).sum()
        return cov / np.sqrt(var_true * var_pred) if var_true * var_pred > 0 else 0

    def train(self, df_target, df_4h_momentum, df_7d_momentum, df_amount_sum, df_vol_momentum, df_atr, df_macd, df_buy_pressure):
        factor1_long = df_4h_momentum.stack()
        factor2_long = df_7d_momentum.stack()
        factor3_long = df_amount_sum.stack()
        factor4_long = df_vol_momentum.stack()
        factor5_long = df_atr.stack()
        factor6_long = df_macd.stack()
        factor7_long = df_buy_pressure.stack()
        target_long = df_target.stack()

        factor1_long.name = '4h_momentum'
        factor2_long.name = '7d_momentum'
        factor3_long.name = 'amount_sum'
        factor4_long.name = 'vol_momentum'
        factor5_long.name = 'atr'
        factor6_long.name = 'macd'
        factor7_long.name = 'buy_pressure'
        target_long.name = 'target'

        data = pd.concat([factor1_long, factor2_long, factor3_long, factor4_long, factor5_long, factor6_long, factor7_long, target_long], axis=1)
        print(f"Data size before dropna: {len(data)}")
        data = data.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
        print(f"Data size after dropna: {len(data)}")

        X = data[['4h_momentum', '7d_momentum', 'amount_sum', 'vol_momentum', 'atr', 'macd', 'buy_pressure']]
        y = data['target'].replace([np.inf, -np.inf], 0)

        X_scaled = self.scaler.fit_transform(X) # stadardize features by z-score 

        tscv = TimeSeriesSplit(n_splits=5)
        best_score = -np.inf
        best_model = None

        for train_idx, val_idx in tscv.split(X_scaled):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            y_train_clean = y_train.fillna(0)
            sample_weight = np.where((y_train_clean > y_train_clean.quantile(0.9)) | (y_train_clean < y_train_clean.quantile(0.1)), 2, 1)

            model = xgb.XGBRegressor(
                objective='reg:squarederror',
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                n_estimators=200,
                early_stopping_rounds=10,
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
        data['y_pred'] = data['y_pred'].replace([np.inf, -np.inf], 0).fillna(0)
        data['y_pred'] = data['y_pred'].ewm(span=5).mean()

        df_submit = data.reset_index(level=0)
        df_submit = df_submit[['level_0', 'y_pred']]
        df_submit['symbol'] = df_submit.index.values
        df_submit = df_submit[['level_0', 'symbol', 'y_pred']]
        df_submit.columns = ['datetime', 'symbol', 'predict_return']
        df_submit = df_submit[df_submit['datetime'] >= self.start_datetime]
        df_submit["id"] = df_submit["datetime"].dt.strftime("%Y%m%d%H%M%S") + "_" + df_submit["symbol"].astype(str)
        df_submit = df_submit[['id', 'predict_return']]

        if os.path.exists(self.submission_id_path):
            df_submission_id = pd.read_csv(self.submission_id_path)
            id_list = df_submission_id["id"].tolist()
            print(f"Submission ID count: {len(id_list)}")
            df_submit_competion = df_submit[df_submit['id'].isin(id_list)]
            missing_elements = list(set(id_list) - set(df_submit_competion['id']))
            print(f"Missing IDs: {len(missing_elements)}")
            new_rows = pd.DataFrame({'id': missing_elements, 'predict_return': [0] * len(missing_elements)})
            df_submit_competion = pd.concat([df_submit_competion, new_rows], ignore_index=True)
        else:
            print(f"Warning: {self.submission_id_path} not found. Saving submission without ID filtering.")
            df_submit_competion = df_submit

        print("Submission file sample:", df_submit_competion.head())
        df_submit_competion.to_csv("submit.csv", index=False)

        df_check = data.reset_index(level=0)
        df_check = df_check[['level_0', 'target']]
        df_check['symbol'] = df_check.index.values
        df_check = df_check[['level_0', 'symbol', 'target']]
        df_check.columns = ['datetime', 'symbol', 'true_return']
        df_check = df_check[df_check['datetime'] >= self.start_datetime]
        df_check["id"] = df_check["datetime"].dt.strftime("%Y%m%d%H%M%S") + "_" + df_check["symbol"]
        df_check = df_check[['id', 'true_return']]
        df_check.to_csv("check.csv", index=False)

        rho_overall = self.weighted_spearmanr(data['target'], data['y_pred'])
        print(f"Weighted Spearman correlation coefficient: {rho_overall:.4f}")

        # SHAP plot
        explainer = shap.Explainer(best_model)
        shap_values = explainer(X_scaled)
        shap.summary_plot(shap_values, X.columns)

    def run(self):
        all_symbol_list, time_arr, vwap_arr, amount_arr, atr_arr, macd_arr, buy_volume_arr, volume_arr = self.get_all_symbol_kline()
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

        windows_1d = 4 * 24 * 1
        windows_7d = 4 * 24 * 7
        windows_4h = 4 * 4

        # calculate short-term and long-term momentum
        df_4h_momentum = (df_vwap / df_vwap.shift(windows_4h) - 1).replace([np.inf, -np.inf], np.nan).fillna(0)
        df_7d_momentum = (df_vwap / df_vwap.shift(windows_7d) - 1).replace([np.inf, -np.inf], np.nan).fillna(0)

        # volume factor
        df_amount_sum = df_amount.rolling(windows_7d).sum().replace([np.inf, -np.inf], np.nan).fillna(0)
        df_vol_momentum = (df_amount / df_amount.shift(windows_1d) - 1).replace([np.inf, -np.inf], np.nan).fillna(0)

        # buy pressure
        df_buy_pressure = (df_buy_volume - (df_volume - df_buy_volume)).replace([np.inf, -np.inf], np.nan).fillna(0)

        # 24 hour return
        df_24hour_rtn = (df_vwap / df_vwap.shift(windows_1d) - 1).replace([np.inf, -np.inf], np.nan).fillna(0)

        # TODO: add more factors
        
        self.train(df_24hour_rtn.shift(-windows_1d), df_4h_momentum, df_7d_momentum, df_amount_sum, df_vol_momentum, df_atr, df_macd, df_buy_pressure)

if __name__ == '__main__':
    # mp.set_start_method('spawn', force=True)
    
    print("Input directory contents:", os.listdir("/kaggle/input/avenir-hku-web/"))
    print("Train data directory contents:", os.listdir("/kaggle/input/avenir-hku-web/kline_data/train_data"))
    model = OptimizedModel()
    model.run()