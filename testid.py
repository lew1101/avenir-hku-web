import pandas as pd

df = pd.read_parquet("kline_data/train_data/ADAUSDT.parquet")
print(f"Raw data for ADAUSDT: {df.head().to_string()}")