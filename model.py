from operator import index
import numpy as np
import pandas as pd  # type: ignore
import datetime, os, time
import torch
import torch.multiprocessing as mp
import xgboost as xgb  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
from sklearn.model_selection import TimeSeriesSplit  # type: ignore
import shap  # type: ignore
import copy

PROCESSES = mp.cpu_count() // 2  # use half of the available CPU cores
BASE_DIR = os.getcwd()
TRAIN_DATA_DIR = os.path.join(BASE_DIR, "kline_data", "train_data")
SUBMISSION_ID_PATH = os.path.join(BASE_DIR, "submission_id.csv")
CACHE_DIR = os.path.join(BASE_DIR, "data_cache")


def compute_factors_torch(df, device):
    # 转换为张量 convert to tensor
    close = torch.tensor(df['close_price'].values, dtype=torch.float32, device=device)
    volume = torch.tensor(df['volume'].values, dtype=torch.float32, device=device)
    amount = torch.tensor(df['amount'].values, dtype=torch.float32, device=device)
    high = torch.tensor(df['high_price'].values, dtype=torch.float32, device=device)
    low = torch.tensor(df['low_price'].values, dtype=torch.float32, device=device)
    buy_volume = torch.tensor(df['buy_volume'].values, dtype=torch.float32, device=device)
    
    def apply_pool(tensor, window, device):
        max_pad = window // 2
        input_length = len(tensor)
        padded = torch.nn.functional.avg_pool1d(tensor.unsqueeze(0), kernel_size=window, stride=1, padding=max_pad).squeeze()
        pad_size = input_length - len(padded)
        if pad_size > 0:
            padded = torch.nn.functional.pad(padded, (0, pad_size), mode='replicate')
        elif pad_size < 0:
            padded = padded[:input_length]
        return padded[:input_length]

    # VWAP - Volume-Weighted Average Price: https://www.investopedia.com/terms/v/vwap.asp
    ## Average price of the security has traded at throughout the day weighted by volume
    vwap = torch.where(volume > 0, amount / volume, close)
    vwap = torch.where(torch.isfinite(vwap), vwap, close)

    # RSI - Relative Strength Index (Full Implementation, 14 day window): https://www.investopedia.com/terms/r/rsi.asp
    ## Momentum indicator to measure how fast & how much a stock moved recently
    ## RSI > 70 -> Asset is overbought, possible trend reversal
    ## RSI < 30 -> Asset maybe oversold, possible reversal outward
    delta = torch.diff(close, prepend=close[:1])  # calculate delta between consecutive elements
    gain = torch.where(delta > 0, delta, torch.tensor(0.0, device=device))
    loss = torch.where(delta < 0, -delta, torch.tensor(0.0, device=device))

    init_gain = gain[:14].mean() if len(gain) > 14 else torch.tensor(0.0, device=device)
    init_loss = loss[:14].mean() if len(gain) > 14 else torch.tensor(0.0, device=device)

    avg_gain = torch.zeros_like(close, device=device)
    avg_loss = torch.zeros_like(close, device=device)
    avg_gain[:14] = init_gain
    avg_loss[:14] = init_loss
    for i in range(14, len(
            close)):  # Wilder's smoothing method -> less reactive than standard ema, more stable
        avg_gain[i] = (avg_gain[i - 1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i - 1] * 13 + loss[i]) / 14

    rs = torch.where(avg_loss > 0, avg_gain / avg_loss, torch.tensor(0.0, device=device))
    rsi = 100 - 100 / (1 + rs)
    rsi = torch.where(torch.isnan(rsi), torch.tensor(50.0, device=device),
                      rsi)  # set to neutral if nan

    # ATR - Average True Range: https://www.investopedia.com/terms/a/atr.asp
    ## Measure of Volatility
    ## Price moves a lot -> High ATR, Price mostly flat -> Low ATR
    tr = torch.max(high - low, torch.max(torch.abs(high - close), torch.abs(low - close)))
    atr = torch.zeros_like(tr, device=device)
    for i in range(14, len(tr)):  # Wilder's
        atr[i] = (atr[i - 1] * 13 + tr[i]) / 14
    atr = torch.where(torch.isnan(atr), torch.tensor(0.0, device=device),
                      atr)  # set to neutral if nan



    # MACD - Moving Average Convergence Divergence: https://www.investopedia.com/terms/m/macd.asp
    ## Momentum + trend-following indicator widely used to spot trade direction, strength, as well as buy/sell signals
    ema12 = torch.zeros_like(close, device=device)
    ema26 = torch.zeros_like(close, device=device)
    alpha12 = 2 / (12 + 1)
    alpha26 = 2 / (26 + 1)
    ema12[:12] = close[:12].mean()
    ema26[:26] = close[:26].mean()
    for i in range(12, len(close)):  # short term ema
        ema12[i] = alpha12 * close[i] + (1 - alpha12) * ema12[i - 1]
    for i in range(26, len(close)):  # long term ema
        ema26[i] = alpha26 * close[i] + (1 - alpha26) * ema26[i - 1]
    macd = ema12 - ema26
    macd = torch.where(torch.isnan(macd), torch.tensor(0.0, device=device), macd)

    # Keltner Channels 
    ## Volatility-based bands around a moving average
    ## Upper Band = EMA + (ATR * Multiplier)
    ## Lower Band = EMA - (ATR * Multiplier)
    ## Commonly used multiplier is 2
    keltner_multiplier = 2
    keltner_upper = ema12 + (atr * keltner_multiplier)
    keltner_lower = ema12 - (atr * keltner_multiplier)
    keltner_upper = torch.where(torch.isfinite(keltner_upper), keltner_upper, close)
    keltner_lower = torch.where(torch.isfinite(keltner_lower), keltner_lower, close)
    # Remove manual padding since tensors should already have correct length
    # keltner_upper = torch.nn.functional.pad(keltner_upper, (19, 0), mode='constant', value=0)
    # keltner_lower = torch.nn.functional.pad(keltner_lower, (19, 0), mode='constant', value=0)
    
    
    # Buy Ratio
    ## Estimate of how much total trading volume is made up of buying pressure vs. selling
    ## Buy Volume: trades that happened at or near ask price -> if relatively high, could signal upward momentum
    ## Sell Volume: trades that happed at or near bid price -> if relatively high, could signal downward momentum
    buy_ratio = torch.where(volume > 0, buy_volume / volume, torch.tensor(0.5, device=device))

    # Bollinger Bands - https://www.investopedia.com/terms/b/bollingerbands.asp
    ## Volatility bands around a moving average
    rolling_mean = apply_pool(close, 20, device)
    # Calculate squared differences and apply the same pooling
    squared_diff = (close - rolling_mean) ** 2
    rolling_var = apply_pool(squared_diff, 20, device)
    rolling_std = torch.sqrt(rolling_var)
    bb_upper = rolling_mean + 2 * rolling_std
    bb_lower = rolling_mean - 2 * rolling_std
    # Remove the manual padding since apply_pool already handles it
    # bb_upper = torch.nn.functional.pad(bb_upper, (19, 0), mode='constant', value=0)
    # bb_lower = torch.nn.functional.pad(bb_lower, (19, 0), mode='constant', value=0)


    
    # Stochastic Oscillator
    ## Momentum indicator comparing closing price to a range of prices over a period
    ## %K = (Current Close - Lowest Low) / (Highest High - Lowest Low) * 100
    input_length = len(close)
    highest_high = torch.nn.functional.max_pool1d(close.unsqueeze(0), kernel_size=14, stride=1, padding=7).squeeze()
    lowest_low = -torch.nn.functional.max_pool1d(-close.unsqueeze(0), kernel_size=14, stride=1, padding=7).squeeze()
    pad_size = input_length - len(highest_high)
    if pad_size > 0:
        highest_high = torch.nn.functional.pad(highest_high, (0, pad_size), mode='replicate')
        lowest_low = torch.nn.functional.pad(lowest_low, (0, pad_size), mode='replicate')
    elif pad_size < 0:
        highest_high = highest_high[:input_length]
        lowest_low = lowest_low[:input_length]
    stochastic_k = (close - lowest_low) / (highest_high - lowest_low + 1e-8) * 100
    stochastic_k = torch.where(torch.isfinite(stochastic_k), stochastic_k, torch.tensor(50.0, device=device))
    stochastic_k = torch.clamp(stochastic_k, 0, 100)
    stochastic_d = apply_pool(stochastic_k, 3, device)
    stochastic_d = torch.clamp(stochastic_d, 0, 100)
    

    

    
    # CCI - Commodity Channel Index: https://www.investopedia.com/terms/c/cci.asp
    ## Measures deviation of price from its average price over a period
    ## CCI = (Typical Price - SMA(Typical Price)) / (0.015 * Mean Deviation)
    typical_price = (high + low + close) / 3
    sma_typical_price = apply_pool(typical_price, 20, device)
    mean_deviation = apply_pool(torch.abs(typical_price - sma_typical_price), 20, device)
    cci = (typical_price - sma_typical_price) / (0.015 * mean_deviation + 1e-8)
    cci = torch.where(torch.isfinite(cci), cci, torch.tensor(0.0, device=device))
    cci = torch.clamp(cci, -100, 100)
    
    # Chaikin Money Flow Index (MFI): https://www.investopedia.com/terms/m/mfi.asp
    ## Measures buying and selling pressure over a period
    ## MFI = (Sum of Positive Money Flow - Sum of Negative Money Flow) / (Sum of Positive Money Flow + Sum of Negative Money Flow)
    money_flow = close * volume
    positive_money_flow = torch.where(close[1:] > close[:-1], money_flow[1:], torch.tensor(0.0, device=device))
    negative_money_flow = torch.where(close[1:] < close[:-1], money_flow[1:], torch.tensor(0.0, device=device))
    sum_positive_flow = apply_pool(torch.cat([torch.tensor([0.0], device=device), positive_money_flow]), 20, device)
    sum_negative_flow = apply_pool(torch.cat([torch.tensor([0.0], device=device), negative_money_flow]), 20, device)
    mfi = (sum_positive_flow - sum_negative_flow) / (sum_positive_flow + sum_negative_flow + 1e-8)
    mfi = torch.where(torch.isfinite(mfi), mfi, torch.tensor(0.0, device=device))
    mfi = torch.clamp(mfi, -1, 1)
    
    
    #obv - On-Balance Volume: https://www.investopedia.com/terms/o/onbalancevolume.asp
    ## Volume-based indicator that uses volume flow to predict price changes
    obv = torch.zeros_like(close, device=device)
    for i in range(1, len(close)):
        obv[i] = obv[i-1] + torch.where(close[i] > close[i-1], volume[i], 
                                        torch.where(close[i] < close[i-1], -volume[i], torch.tensor(0.0, device=device)))
    obv = torch.where(torch.isfinite(obv), obv, torch.tensor(0.0, device=device))
    obv = torch.clamp(obv, -1e6, 1e6)
    
    
    
    # VWAP Deviation
    ## Measure of how far current price is from VWAP
    ## Price > VWAP - Buyers pushing up price -> potentially overbought
    ## Price < VWAP - Sellers pushing down price -> potentially oversold
    ## Large deviations can signal mean reversion
    vwap_deviation = (close - vwap) / torch.where(vwap != 0, vwap, torch.tensor(1.0, device=device))
    vwap_deviation = torch.where(torch.isfinite(vwap_deviation), vwap_deviation,
                                 torch.tensor(0.0, device=device))

    # Convert torch tensors to dataframes
    df['vwap'] = vwap.cpu().numpy()
    df['rsi'] = rsi.cpu().numpy()
    df['macd'] = macd.cpu().numpy()
    df['atr'] = atr.cpu().numpy()
    df['buy_ratio'] = buy_ratio.cpu().numpy()
    df['vwap_deviation'] = vwap_deviation.cpu().numpy()
    df['keltner_upper'] = keltner_upper.cpu().numpy()
    df['keltner_lower'] = keltner_lower.cpu().numpy()
    df['stochastic_d'] = stochastic_d.cpu().numpy()
    df['cci'] = cci.cpu().numpy()
    df['mfi'] = mfi.cpu().numpy()
    df['obv'] = obv.cpu().numpy()
    df['bb_upper'] = bb_upper.cpu().numpy()
    df['bb_lower'] = bb_lower.cpu().numpy()

    
    return df


def get_single_symbol_kline_data(symbol, train_data_path, device):
    try:
        df = pd.read_parquet(f"{train_data_path}/{symbol}.parquet")
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp')
        df = df.astype(np.float64)
        required_cols = [
            'open_price', 'high_price', 'low_price', 'close_price', 'volume', 'amount', 'buy_volume'
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"{symbol} missing columns: {missing_cols}")
            return pd.DataFrame(columns=required_cols +
                                ['vwap', 'rsi', 'macd', 'buy_ratio', 'vwap_deviation', 'atr'])

        # smooth data by removing extreme outliers
        df['close_price'] = df['close_price'].clip(df['close_price'].quantile(0.01),
                                                   df['close_price'].quantile(0.99))
        df['volume'] = df['volume'].clip(df['volume'].quantile(0.01), df['volume'].quantile(0.99))

        # Calculate the indicators
        df = compute_factors_torch(df, device)
        print(f"Loaded data for {symbol}, shape: {df.shape}, vwap NaNs: {df['vwap'].isna().sum()}")
        return df
    except Exception as e:
        print(f"Error loading {symbol}: {e}")
        return pd.DataFrame(columns=[
            'open_price', 'high_price', 'low_price', 'close_price', 'volume', 'amount',
            'buy_volume', 'vwap', 'rsi', 'macd', 'buy_ratio', 'vwap_deviation', 'atr'
        ])


class OptimizedModel:

    def __init__(self):
        self.train_data_path = TRAIN_DATA_DIR
        self.submission_id_path = SUBMISSION_ID_PATH
        self.start_datetime = datetime.datetime(2021, 3, 1, 0, 0, 0)
        self.scaler = StandardScaler()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.device = 'cpu'
        self.data_cache = {}
        print(f"Using device: {self.device}")

    def get_all_symbol_list(self):
        try:
            parquet_name_list = os.listdir(self.train_data_path)
            symbol_list = [parquet_name.split(".")[0] for parquet_name in parquet_name_list]
            return symbol_list
        except Exception as e:
            print(f"Error in get_all_symbol_list: {e}")
            return []

    def get_all_symbol_kline(self):
        t0 = datetime.datetime.now()

        try:
            pool = mp.Pool(processes=PROCESSES)  # use two cores

            all_symbol_list = self.get_all_symbol_list()
            if not all_symbol_list:
                print("No symbols found, exiting.")
                pool.close()
                return [], [], [], [], [], [], [], [], [], []

            # get symbol data + indicators
            df_list = [
                pool.apply_async(get_single_symbol_kline_data,
                                 (symbol, self.train_data_path, self.device))
                for symbol in all_symbol_list
            ]
        except KeyboardInterrupt as e:
            pool.terminate()
            raise

        finally:
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
        time_index = pd.date_range(start=self.start_datetime, end='2024-12-31',
                                   freq='15min')  # 15 min time index
        df_results = [async_result.get() for async_result in df_list]
        df_open_price = pd.concat([
            result['open_price']
            for result in df_results
            if not result.empty and 'open_price' in result.columns
        ],
                                  axis=1).sort_index(
                                      ascending=True)  # get open price dfs for each symbol
        print(
            f"df_open_price index dtype: {df_open_price.index.dtype}, shape: {df_open_price.shape}")
        df_open_price.columns = loaded_symbols
        df_open_price = df_open_price.reindex(columns=all_symbol_list, fill_value=0).reindex(
            time_index, method='ffill')  # unify time steps
        time_arr = pd.to_datetime(df_open_price.index).values

        def align_df(arr, valid_symbols, key):
            valid_dfs = [
                df[key]
                for df, s in zip([i.get() for i in df_list], all_symbol_list)
                if not df.empty and key in df.columns and s in valid_symbols
            ]
            if not valid_dfs:
                print(f"No valid data for {key}, filling with zeros")
                return np.zeros((len(time_index), len(all_symbol_list)))
            df = pd.concat(valid_dfs, axis=1).sort_index(ascending=True)
            df.columns = valid_symbols
            return df.reindex(columns=all_symbol_list, fill_value=0).reindex(time_index,
                                                                             method='ffill').values

        vwap_arr = align_df(df_list, loaded_symbols, 'vwap')
        amount_arr = align_df(df_list, loaded_symbols, 'amount')
        atr_arr = align_df(df_list, loaded_symbols, 'atr')
        macd_arr = align_df(df_list, loaded_symbols, 'macd')
        buy_volume_arr = align_df(df_list, loaded_symbols, 'buy_volume')
        volume_arr = align_df(df_list, loaded_symbols, 'volume')

        print(f"Finished get all symbols kline, time elapsed: {datetime.datetime.now() - t0}")
        return all_symbol_list, time_arr, vwap_arr, amount_arr, atr_arr, macd_arr, buy_volume_arr, volume_arr
    
    def superior_get_all_symbol_kline(self):
        t0 = datetime.datetime.now()

        try:
            pool = mp.Pool(processes=PROCESSES)  # use two cores

            all_symbol_list = self.get_all_symbol_list()
            if not all_symbol_list:
                print("No symbols found, exiting.")
                pool.close()
                return [], [], [], [], [], [], [], [], [], []

            # get symbol data + indicators
            df_list = [
                pool.apply_async(get_single_symbol_kline_data,
                                 (symbol, self.train_data_path, self.device))
                for symbol in all_symbol_list
            ]
        except KeyboardInterrupt as e:
            pool.terminate()
            raise

        finally:
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
        time_index = pd.date_range(start=self.start_datetime, end='2024-12-31',
                                   freq='15min')  # 15 min time index
        df_results = [async_result.get() for async_result in df_list]
        df_open_price = pd.concat([
            result['open_price']
            for result in df_results
            if not result.empty and 'open_price' in result.columns
        ],
                                  axis=1).sort_index(
                                      ascending=True)  # get open price dfs for each symbol
        print(
            f"df_open_price index dtype: {df_open_price.index.dtype}, shape: {df_open_price.shape}")
        df_open_price.columns = loaded_symbols
        df_open_price = df_open_price.reindex(columns=all_symbol_list, fill_value=0).reindex(
            time_index, method='ffill')  # unify time steps
        time_arr = pd.to_datetime(df_open_price.index).values

        def align_df(arr, valid_symbols, key):
            valid_dfs = [
                df[key]
                for df, s in zip([i.get() for i in df_list], all_symbol_list)
                if not df.empty and key in df.columns and s in valid_symbols
            ]
            if not valid_dfs:
                print(f"No valid data for {key}, filling with zeros")
                return np.zeros((len(time_index), len(all_symbol_list)))
            df = pd.concat(valid_dfs, axis=1).sort_index(ascending=True)
            df.columns = valid_symbols
            return df.reindex(columns=all_symbol_list, fill_value=0).reindex(time_index,
                                                                             method='ffill').values

        vwap_arr = align_df(df_list, loaded_symbols, 'vwap')
        amount_arr = align_df(df_list, loaded_symbols, 'amount')
        atr_arr = align_df(df_list, loaded_symbols, 'atr')
        macd_arr = align_df(df_list, loaded_symbols, 'macd')
        buy_volume_arr = align_df(df_list, loaded_symbols, 'buy_volume')
        volume_arr = align_df(df_list, loaded_symbols, 'volume')
        rsi_arr = align_df(df_list, loaded_symbols, 'rsi')
        vwap_deviation_arr = align_df(df_list, loaded_symbols, 'vwap_deviation')
        bb_upper_arr = align_df(df_list, loaded_symbols, 'bb_upper')
        bb_lower_arr = align_df(df_list, loaded_symbols, 'bb_lower')
        keltner_upper_arr = align_df(df_list, loaded_symbols, 'keltner_upper')
        keltner_lower_arr = align_df(df_list, loaded_symbols, 'keltner_lower')  
        stochastic_d_arr = align_df(df_list, loaded_symbols, 'stochastic_d')
        cci_arr = align_df(df_list, loaded_symbols, 'cci')
        mfi_arr = align_df(df_list, loaded_symbols, 'mfi')
        obv_arr = align_df(df_list, loaded_symbols, 'obv')
        

        print(f"Finished get all symbols kline, time elapsed: {datetime.datetime.now() - t0}")
        return all_symbol_list, time_arr, vwap_arr, amount_arr, atr_arr, macd_arr, buy_volume_arr, volume_arr, rsi_arr, vwap_deviation_arr, bb_upper_arr, bb_lower_arr, keltner_upper_arr, keltner_lower_arr, stochastic_d_arr, cci_arr, mfi_arr, obv_arr


    def weighted_spearmanr(self, y_true, y_pred):
        n = len(y_true)
        r_true = pd.Series(y_true).rank(ascending=False, method='average')
        r_pred = pd.Series(y_pred, index = y_true.index).rank(ascending=False, method='average') # Only change
        x = 2 * (r_true - 1) / (n - 1) - 1
        w = x**2
        w_sum = w.sum()
        mu_true = (w * r_true).sum() / w_sum
        mu_pred = (w * r_pred).sum() / w_sum
        cov = (w * (r_true - mu_true) * (r_pred - mu_pred)).sum()
        var_true = (w * (r_true - mu_true)**2).sum()
        var_pred = (w * (r_pred - mu_pred)**2).sum()
        return cov / np.sqrt(var_true * var_pred) if var_true * var_pred > 0 else 0

    def train(self, df_target, df_4h_momentum, df_7d_momentum, df_amount_sum, df_vol_momentum,
              df_atr, df_macd, df_buy_pressure, df_rsi, df_vwapdeviation, df_1h_momentum, df_bb_upper, df_bb_lower,
              df_keltner_upper, df_keltner_lower, df_stochastic_d, df_cci, df_mfi, df_obv):
        factor1_long = df_4h_momentum.stack()
        factor2_long = df_7d_momentum.stack()
        factor3_long = df_amount_sum.stack()
        factor4_long = df_vol_momentum.stack()
        factor5_long = df_atr.stack()
        factor6_long = df_macd.stack()
        factor7_long = df_buy_pressure.stack()
        factor8_long = df_rsi.stack()
        factor9_long = df_vwapdeviation.stack()
        factor10_long = df_1h_momentum.stack()
        factor11_long = df_bb_upper.stack()
        factor12_long = df_bb_lower.stack()
        factor13_long = df_keltner_upper.stack()
        factor14_long = df_keltner_lower.stack()
        factor15_long = df_stochastic_d.stack()
        factor16_long = df_cci.stack()
        factor17_long = df_mfi.stack()
        factor18_long = df_obv.stack()


        target_long = df_target.stack()
        

        factor1_long.name = '4h_momentum'
        factor2_long.name = '7d_momentum'
        factor3_long.name = 'amount_sum'
        factor4_long.name = 'vol_momentum'
        factor5_long.name = 'atr'
        factor6_long.name = 'macd'
        factor7_long.name = 'buy_pressure'
        factor8_long.name = 'rsi'
        factor9_long.name = 'vwap_deviation'
        factor10_long.name = '1h_momentum'
        factor11_long.name = 'bb_upper'
        factor12_long.name = 'bb_lower'
        factor13_long.name = 'keltner_upper'
        factor14_long.name = 'keltner_lower'
        factor15_long.name = 'stochastic_d'
        factor16_long.name = 'cci'
        factor17_long.name = 'mfi'
        factor18_long.name = 'obv'

        target_long.name = 'target'

        data = pd.concat([
            factor1_long, factor2_long, factor3_long, factor4_long, factor5_long, factor6_long,
            factor7_long, factor8_long, factor9_long, factor10_long, factor11_long, factor12_long,
            factor13_long, factor14_long, factor15_long, factor16_long, factor17_long, factor18_long,
            target_long
        ],
                         axis=1)

        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.dropna()
        


        X = data[[
            '4h_momentum', '7d_momentum', 'amount_sum', 'vol_momentum', 'atr', 'macd',
            'buy_pressure', 'rsi', 'vwap_deviation', '1h_momentum', 'keltner_upper',
            'keltner_lower', 'stochastic_d', 'cci', 'mfi', 'obv', 'bb_upper', 'bb_lower'
        ]]
        y = data['target'].replace([np.inf, -np.inf], 0)

        X_scaled = self.scaler.fit_transform(X)  # stadardize features by z-score

        tscv = TimeSeriesSplit(n_splits=5)
        best_score = -np.inf
        best_model = None

        for train_idx, val_idx in tscv.split(X_scaled):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            y_train_clean = y_train.fillna(0)
            sample_weight = np.where((y_train_clean > y_train_clean.quantile(0.9)) |
                                     (y_train_clean < y_train_clean.quantile(0.1)), 2, 1)

            model = xgb.XGBRegressor(objective='reg:squarederror',
                                     learning_rate=0.01,
                                     max_depth=20,
                                     subsample=0.8,
                                     n_estimators=500,
                                     reg_lambda=1,
                                     tree_method='hist',
                                     device = 'cuda',
                                     early_stopping_rounds=20,
                                     random_state=42)
            model.fit(X_train,
                      y_train,
                      sample_weight=sample_weight,
                      eval_set=[(X_val, y_val)],
                      verbose=False)

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
        df_submit["id"] = df_submit["datetime"].dt.strftime(
            "%Y-%m-%d %H:%M:%S") + "_" + df_submit["symbol"]
        df_submit = df_submit[['id', 'predict_return']]

        if os.path.exists(self.submission_id_path):
            df_submission_id = pd.read_csv(self.submission_id_path)
            # print("Submission ID sample:", df_submission_id.head())
            id_list = df_submission_id["id"].tolist()
            print(f"Submission ID count: {len(id_list)}")
            df_submit_competion = df_submit[df_submit['id'].isin(id_list)]
            missing_elements = list(set(id_list) - set(df_submit_competion['id']))
            print(f"Missing IDs: {len(missing_elements)}")
            new_rows = pd.DataFrame({
                'id': missing_elements,
                'predict_return': [0] * len(missing_elements)
            })
            df_submit_competion = pd.concat([df_submit_competion, new_rows], ignore_index=True)
        else:
            print(
                f"Warning: {self.submission_id_path} not found. Saving submission without ID filtering."
            )
            df_submit_competion = df_submit

        print("Submission file sample:", df_submit_competion.head())
        df_submit_competion.to_csv("submit.csv", index=False)

        df_check = data.reset_index(level=0)
        df_check = df_check[['level_0', 'target']]
        df_check['symbol'] = df_check.index.values
        df_check = df_check[['level_0', 'symbol', 'target']]
        df_check.columns = ['datetime', 'symbol', 'true_return']
        df_check = df_check[df_check['datetime'] >= self.start_datetime]
        df_check["id"] = df_check["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S") + "_" + df_check["symbol"]
        df_check = df_check[['id', 'true_return']]
        df_check.to_csv("check.csv", index=False)

        rho_overall = self.weighted_spearmanr(data['target'], data['y_pred'])
        print(f"Weighted Spearman correlation coefficient: {rho_overall:.4f}")

        # SHAP plot
        explainer = shap.Explainer(best_model)
        shap_values = explainer(X_scaled)
        shap.summary_plot(shap_values, X.columns)

    def run(self):
        all_symbol_list, time_arr, vwap_arr, amount_arr, atr_arr, macd_arr, buy_volume_arr, volume_arr, rsi_arr, vwap_deviation_arr, bb_upper_arr, bb_lower_arr, keltner_upper_arr, keltner_lower_arr, stochastic_d_arr, cci_arr, mfi_arr, obv_arr = self.superior_get_all_symbol_kline(
        )
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
        df_vwapdeviation = pd.DataFrame(
            vwap_deviation_arr, columns=all_symbol_list, index=time_arr)
        df_rsi = pd.DataFrame(rsi_arr, columns=all_symbol_list, index=time_arr)
        df_bb_upper = pd.DataFrame(bb_upper_arr, columns=all_symbol_list, index=time_arr)
        df_bb_lower = pd.DataFrame(bb_lower_arr, columns=all_symbol_list, index=time_arr)
        df_keltner_upper = pd.DataFrame(keltner_upper_arr, columns=all_symbol_list, index=time_arr)
        df_keltner_lower = pd.DataFrame(keltner_lower_arr, columns=all_symbol_list, index=time_arr)
        df_stochastic_d = pd.DataFrame(stochastic_d_arr, columns=all_symbol_list, index=time_arr)
        df_cci = pd.DataFrame(cci_arr, columns=all_symbol_list, index=time_arr)
        df_mfi = pd.DataFrame(mfi_arr, columns=all_symbol_list, index=time_arr)
        df_obv = pd.DataFrame(obv_arr, columns=all_symbol_list, index=time_arr)


        windows_1d = 4 * 24 * 1
        windows_7d = 4 * 24 * 7
        windows_4h = 4 * 4
        windows_1h = 4
        # calculate short-term and long-term momentum
        df_1h_momentum = (df_vwap / df_vwap.shift(windows_1h) - 1).replace([np.inf, -np.inf],
                                                                           np.nan).fillna(0)
        df_4h_momentum = (df_vwap / df_vwap.shift(windows_4h) - 1).replace([np.inf, -np.inf],
                                                                           np.nan).fillna(0)
        df_7d_momentum = (df_vwap / df_vwap.shift(windows_7d) - 1).replace([np.inf, -np.inf],
                                                                           np.nan).fillna(0)

        # Could add more momentum features here, like 10hr, 1d, 2d, 3d, etc.
        
        # volume factor
        df_amount_sum = df_amount.rolling(windows_7d).sum().replace([np.inf, -np.inf],
                                                                    np.nan).fillna(0)
        df_vol_momentum = (df_amount / df_amount.shift(windows_1d) - 1).replace([np.inf, -np.inf],
                                                                                np.nan).fillna(0)

        # buy pressure
        df_buy_pressure = (df_buy_volume - (df_volume - df_buy_volume)).replace([np.inf, -np.inf],
                                                                                np.nan).fillna(0)

        # 24 hour return
        df_24hour_rtn = (df_vwap / df_vwap.shift(windows_1d) - 1).replace([np.inf, -np.inf],
                                                                          np.nan).fillna(0)



        # TODO: add more factors

        # Compare against btc, btc usually has the highest volume and liquidity // might need to import new data

        # 

        self.train(df_24hour_rtn.shift(-windows_1d), df_4h_momentum, df_7d_momentum, df_amount_sum,
                   df_vol_momentum, df_atr, df_macd, df_buy_pressure, df_rsi, df_vwapdeviation, df_1h_momentum, df_bb_upper,df_bb_lower, df_keltner_upper, df_keltner_lower, df_stochastic_d, df_cci, df_mfi, df_obv)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    print("Train data directory contents:", os.listdir(TRAIN_DATA_DIR))
    model = OptimizedModel()
    model.run()
