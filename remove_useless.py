import os
import pandas as pd

def get_timestamp_ranges(dfs):
    ranges = {}
    for name, df in dfs.items():
        idx = pd.to_datetime(df.index)
        if idx.min() != pd.Timestamp("2025-01-01 00:00:00") or idx.max() != pd.Timestamp("2025-07-01 02:15:00"):
            ranges[name] = (idx.min(), idx.max())
        else:
            continue
    return pd.DataFrame(ranges, index=["start", "end"]).T

def remove_invalid_parquet_files(folder_path):
    # Load all .parquet files into a dictionary
    dfs = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".parquet"):
            symbol = filename.replace(".parquet", "")
            try:
                df = pd.read_parquet(os.path.join(folder_path, filename))
                dfs[symbol] = df
            except Exception as e:
                print(f"Failed to read {filename}: {e}")

    # Get timestamp ranges
    ranges_df = get_timestamp_ranges(dfs)

    # Identify symbols with NaT in both start and end
    invalid_symbols = ranges_df[ranges_df.isna().all(axis=1)].index.tolist()

    # Remove corresponding files
    for symbol in invalid_symbols:
        file_path = os.path.join(folder_path, f"{symbol}.parquet")
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Removed: {file_path}")
        else:
            print(f"File not found: {file_path}")

# Example usage
remove_invalid_parquet_files("2025_data")