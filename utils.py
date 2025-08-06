import numpy as np
import pandas as pd
import torch


def load_single_symbol_data(args):
    symbol, train_data_path, time_index = args
    try:
        df = pd.read_parquet(f"{train_data_path}/{symbol}.parquet")
        print(f"Raw data for {symbol}: {df.head().to_string()}")
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp')
        df = df.astype(np.float64)
        required_cols = ['open_price', 'high_price', 'low_price', 'close_price', 'volume', 'amount', 'buy_volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"{symbol} missing columns: {missing_cols}")
            return pd.DataFrame(index=time_index,
                                columns=required_cols + ['vwap', 'rsi', 'macd', 'atr', 'bb_upper', 'bb_lower',
                                                         '1h_momentum']).fillna(0).infer_objects(copy=False)
        df['close_price'] = df['close_price'].clip(df['close_price'].quantile(0.01), df['close_price'].quantile(0.99))
        df['volume'] = df['volume'].clip(df['volume'].quantile(0.01), df['volume'].quantile(0.99))

        # 转换为张量并保持在 GPU 上
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        close = torch.tensor(df['close_price'].values, dtype=torch.float32, device=device)
        volume = torch.tensor(df['volume'].values, dtype=torch.float32, device=device)
        amount = torch.tensor(df['amount'].values, dtype=torch.float32, device=device)
        high = torch.tensor(df['high_price'].values, dtype=torch.float32, device=device)
        low = torch.tensor(df['low_price'].values, dtype=torch.float32, device=device)
        buy_volume = torch.tensor(df['buy_volume'].values, dtype=torch.float32, device=device)

        # VWAP
        vwap = torch.where(volume > 0, amount / volume, close)
        mask = ~torch.isfinite(vwap)
        if mask.any():
            vwap = torch.nan_to_num(vwap, nan=0.0)
            n = len(vwap)
            indices = torch.arange(n, device=device).float()
            mask_indices = indices[mask]
            valid_indices = indices[~mask]
            if len(valid_indices) > 1:
                vwap = torch.interp(mask_indices, valid_indices, vwap[~mask])
            vwap = torch.where(torch.isfinite(vwap), vwap, close)

        # RSI (14 周期)
        delta = torch.diff(close, prepend=close[:1])
        gain = torch.where(delta > 0, delta, torch.tensor(0.0, device=device))
        loss = torch.where(delta < 0, -delta, torch.tensor(0.0, device=device))
        init_gain = gain[:14].mean() if len(gain) > 14 else torch.tensor(0.0, device=device)
        init_loss = loss[:14].mean() if len(gain) > 14 else torch.tensor(0.0, device=device)
        avg_gain = torch.zeros_like(close, device=device)
        avg_loss = torch.zeros_like(close, device=device)
        avg_gain[:14] = init_gain
        avg_loss[:14] = init_loss
        for i in range(14, len(close)):
            avg_gain[i] = (avg_gain[i - 1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i - 1] * 13 + loss[i]) / 14
        rs = torch.where(avg_loss > 0, avg_gain / avg_loss, torch.tensor(0.0, device=device))
        rsi = 100 - 100 / (1 + rs)
        rsi = torch.where(torch.isnan(rsi), torch.tensor(50.0, device=device), rsi)

        # MACD
        ema12 = torch.zeros_like(close, device=device)
        ema26 = torch.zeros_like(close, device=device)
        alpha12 = 2 / (12 + 1)
        alpha26 = 2 / (26 + 1)
        ema12[:12] = close[:12].mean()
        ema26[:26] = close[:26].mean()
        for i in range(12, len(close)):
            ema12[i] = alpha12 * close[i] + (1 - alpha12) * ema12[i - 1]
        for i in range(26, len(close)):
            ema26[i] = alpha26 * close[i] + (1 - alpha26) * ema26[i - 1]
        macd = ema12 - ema26
        macd = torch.where(torch.isnan(macd), torch.tensor(0.0, device=device), macd)

        # ATR
        tr = torch.max(high - low, torch.max(torch.abs(high - close), torch.abs(low - close)))
        atr = torch.zeros_like(tr, device=device)
        for i in range(14, len(tr)):
            atr[i] = (atr[i - 1] * 13 + tr[i]) / 14
        atr = torch.where(torch.isnan(atr), torch.tensor(0.0, device=device), atr)

        # Bollinger Bands (20 周期, 2 标准差)
        # Use proper padding to ensure consistent lengths
        padding = 10  # (20 - 1) // 2
        rolling_mean = torch.nn.functional.avg_pool1d(
            torch.nn.functional.pad(close.unsqueeze(0), (padding, padding), mode='replicate'), 
            20, stride=1
        ).squeeze()
        
        # Calculate squared differences with proper padding
        squared_diff = (close - rolling_mean) ** 2
        rolling_var = torch.nn.functional.avg_pool1d(
            torch.nn.functional.pad(squared_diff.unsqueeze(0), (padding, padding), mode='replicate'), 
            20, stride=1
        ).squeeze()
        rolling_std = torch.sqrt(rolling_var)
        
        bb_upper = rolling_mean + 2 * rolling_std
        bb_lower = rolling_mean - 2 * rolling_std
        # Remove manual padding since we already handled it properly above
        # bb_upper = torch.nn.functional.pad(bb_upper, (19, 0), mode='constant', value=0)
        # bb_lower = torch.nn.functional.pad(bb_lower, (19, 0), mode='constant', value=0)

        # # 1 小时动量 (16 周期)
        # momentum_1h = (close / torch.roll(close, 16) - 1).replace([np.inf, -np.inf], np.nan).fillna(0)

        # 转换为 DataFrame 并对齐时间索引
        df = df.reindex(time_index)
        df['vwap'] = torch.interp(torch.arange(len(time_index), device=device).float(),
                                  torch.arange(len(vwap), device=device).float(), vwap).cpu().numpy()
        df['rsi'] = torch.interp(torch.arange(len(time_index), device=device).float(),
                                 torch.arange(len(rsi), device=device).float(), rsi).cpu().numpy()
        df['macd'] = torch.interp(torch.arange(len(time_index), device=device).float(),
                                  torch.arange(len(macd), device=device).float(), macd).cpu().numpy()
        df['atr'] = torch.interp(torch.arange(len(time_index), device=device).float(),
                                 torch.arange(len(atr), device=device).float(), atr).cpu().numpy()
        df['bb_upper'] = torch.interp(torch.arange(len(time_index), device=device).float(),
                                      torch.arange(len(bb_upper), device=device).float(), bb_upper).cpu().numpy()
        df['bb_lower'] = torch.interp(torch.arange(len(time_index), device=device).float(),
                                      torch.arange(len(bb_lower), device=device).float(), bb_lower).cpu().numpy()
        # df['1h_momentum'] = torch.interp(torch.arange(len(time_index), device=device).float(),
        #                                  torch.arange(len(momentum_1h), device=device).float(),
        #                                  momentum_1h).cpu().numpy()
        return df.fillna(0).infer_objects(copy=False)
    except Exception as e:
        print(f"Error loading {symbol}: {e}")
        return pd.DataFrame(index=time_index,
                            columns=['open_price', 'high_price', 'low_price', 'close_price', 'volume', 'amount',
                                     'buy_volume', 'vwap', 'rsi', 'macd', 'atr', 'bb_upper', 'bb_lower',
                                     '1h_momentum']).fillna(0).infer_objects(copy=False)