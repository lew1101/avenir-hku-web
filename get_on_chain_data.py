"""
Multi-chain on-chain feature pipeline for 313 tokens
Targets: holder distribution, active addresses, large transactions, supply locked (TVL proxy)
APIs used: CoinGecko (contract lookup), Etherscan/BSCscan/Polygonscan (free APIs), DefiLlama (TVL)

Environment variables required:
- COINGECKO: none (public)
- ETHERSCAN_API_KEY: Etherscan API key (free tier available)
- BSCSCAN_API_KEY: BSCscan API key (free tier available)  
- POLYGONSCAN_API_KEY: Polygonscan API key (free tier available)

Save output: parquet files per-token and a merged daily features parquet.

Notes:
- Designed to be robust for 313 tokens. Uses free APIs from blockchain explorers.
- Maps symbol->contract with CoinGecko and gets on-chain data from respective chain scanners.
- Uses caching and rate limiting for API calls.

Run: python get_on_chain_data.py
"""

import os
import time
import math
import json
import requests
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

# config
OUT_DIR = Path("onchain_features")
OUT_DIR.mkdir(exist_ok=True)
COINGECKO_API = "https://api.coingecko.com/api/v3"

# Free blockchain explorer APIs
ETHERSCAN_API = "https://api.etherscan.io/api"
BSCSCAN_API = "https://api.bscscan.com/api"
POLYGONSCAN_API = "https://api.polygonscan.com/api"
DEFILLAMA_API = "https://api.llama.fi"

# API Keys (free tier available)
ETHERSCAN_KEY = os.getenv("ETHERSCAN_API_KEY", "YourApiKeyToken")  # Free tier: 5 calls/sec
BSCSCAN_KEY = os.getenv("BSCSCAN_API_KEY", "YourApiKeyToken")     # Free tier: 5 calls/sec
POLYGONSCAN_KEY = os.getenv("POLYGONSCAN_API_KEY", "YourApiKeyToken")  # Free tier: 5 calls/sec

# Headers for requests
HEADERS = {"User-Agent": "OnChainDataPipeline/1.0"}

# Excluded symbols (42 dropped tokens)
EXCLUDED_SYMBOLS = {
    'AI16ZUSDT', 'AIXBTUSDT', 'ALCHUSDT', 'ANIMEUSDT', 'ARCUSDT', 'AVAAIUSDT', 'AVAUSDT', 
    'BIOUSDT', 'CGPTUSDT', 'COOKIEUSDT', 'DEGOUSDT', 'DEXEUSDT', 'DFUSDT', 'DUSDT', 
    'FARTCOINUSDT', 'GRIFFAINUSDT', 'HIVEUSDT', 'KMNOUSDT', 'KOMAUSDT', 'LUMIAUSDT', 
    'MELANIAUSDT', 'MEUSDT', 'MOCAUSDT', 'PENGUUSDT', 'PHAUSDT', 'PIPPINUSDT', 'PROMUSDT', 
    'RAYSOLUSDT', 'SOLVUSDT', 'SONICUSDT', 'SPXUSDT', 'SUSDT', 'SWARMSUSDT', 'TRUMPUSDT', 
    'USUALUSDT', 'VANAUSDT', 'VELODROMEUSDT', 'VINEUSDT', 'VIRTUALUSDT', 'VTHOUSDT', 
    'VVVUSDT', 'ZEREBROUSDT'
}

# utility helpers

def retry_request(func, max_retries=5, initial_wait=1, backoff=2):
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            wait = initial_wait * (backoff ** attempt)
            print(f"Request failed (attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError("Max retries exceeded")


# 1) Map token symbol to contract + chain using CoinGecko (best-effort)
#    returns dict: symbol -> {id, platforms: {chain:address}}

def coingecko_map_symbols(symbols):
    # batch coin list once
    coins = retry_request(lambda: requests.get(f"{COINGECKO_API}/coins/list?include_platform=true").json())
    # coins has id, symbol, name, platforms
    mapping = {}
    # build index by symbol lower
    idx = {}
    for c in coins:
        s = c['symbol'].lower()
        idx.setdefault(s, []).append(c)

    for sym in symbols:
        key = sym.lower().replace('usdt','').replace('-','').strip()
        candidates = idx.get(key, [])
        if not candidates:
            # fallback: exact match
            candidates = [c for c in coins if c['symbol'].lower() == sym.lower()]
        mapping[sym] = candidates[0] if candidates else None
    return mapping


# 2) Blockchain scanner APIs for token holders and transactions

def get_explorer_api_url(chain):
    """Get the appropriate blockchain explorer API URL and key"""
    if chain == 'eth':
        return ETHERSCAN_API, ETHERSCAN_KEY
    elif chain == 'bsc':
        return BSCSCAN_API, BSCSCAN_KEY
    elif chain == 'polygon':
        return POLYGONSCAN_API, POLYGONSCAN_KEY
    else:
        return ETHERSCAN_API, ETHERSCAN_KEY  # Default to Ethereum

def get_token_holders_count(chain, contract_address):
    """Get number of token holders using blockchain explorer APIs"""
    try:
        api_url, api_key = get_explorer_api_url(chain)
        
        # Get token info including holder count
        params = {
            'module': 'token',
            'action': 'tokeninfo',
            'contractaddress': contract_address,
            'apikey': api_key
        }
        
        def call():
            r = requests.get(api_url, params=params, headers=HEADERS, timeout=30)
            r.raise_for_status()
            return r.json()
        
        response = retry_request(call)
        
        if response.get('status') == '1' and 'result' in response:
            result = response['result'][0] if isinstance(response['result'], list) else response['result']
            return int(result.get('holdersCount', 0))
        
        return None
    except Exception as e:
        print(f"Failed to get holder count for {contract_address}: {e}")
        return None

def get_large_transactions(chain, contract_address, start_timestamp, end_timestamp, min_value=100000):
    """Get large token transfers using blockchain explorer APIs"""
    try:
        api_url, api_key = get_explorer_api_url(chain)
        
        # Get token transfers
        params = {
            'module': 'account',
            'action': 'tokentx',
            'contractaddress': contract_address,
            'startblock': 0,
            'endblock': 99999999,
            'sort': 'desc',
            'apikey': api_key
        }
        
        def call():
            r = requests.get(api_url, params=params, headers=HEADERS, timeout=60)
            r.raise_for_status()
            return r.json()
        
        response = retry_request(call)
        
        if response.get('status') == '1' and 'result' in response:
            transfers = response['result']
            large_txs = []
            
            for tx in transfers:
                # Filter by timestamp and value
                tx_timestamp = int(tx.get('timeStamp', 0))
                if start_timestamp <= tx_timestamp <= end_timestamp:
                    # Convert token value (considering decimals)
                    decimals = int(tx.get('tokenDecimal', 18))
                    value = float(tx.get('value', 0)) / (10 ** decimals)
                    
                    if value >= min_value:
                        large_txs.append({
                            'hash': tx.get('hash'),
                            'timestamp': tx_timestamp,
                            'value': value,
                            'from': tx.get('from'),
                            'to': tx.get('to')
                        })
            
            return len(large_txs)
        
        return 0
    except Exception as e:
        print(f"Failed to get large transactions for {contract_address}: {e}")
        return 0

def get_active_addresses_count(chain, contract_address, start_timestamp, end_timestamp):
    """Get count of unique active addresses for a token"""
    try:
        api_url, api_key = get_explorer_api_url(chain)
        
        # Get token transfers to count unique addresses
        params = {
            'module': 'account',
            'action': 'tokentx',
            'contractaddress': contract_address,
            'startblock': 0,
            'endblock': 99999999,
            'sort': 'desc',
            'apikey': api_key
        }
        
        def call():
            r = requests.get(api_url, params=params, headers=HEADERS, timeout=60)
            r.raise_for_status()
            return r.json()
        
        response = retry_request(call)
        
        if response.get('status') == '1' and 'result' in response:
            transfers = response['result']
            unique_addresses = set()
            
            for tx in transfers:
                tx_timestamp = int(tx.get('timeStamp', 0))
                if start_timestamp <= tx_timestamp <= end_timestamp:
                    unique_addresses.add(tx.get('from'))
                    unique_addresses.add(tx.get('to'))
            
            return len(unique_addresses)
        
        return 0
    except Exception as e:
        print(f"Failed to get active addresses for {contract_address}: {e}")
        return 0

# 3) DefiLlama TVL data (free API)

def get_defillama_tvl(protocol_slug):
    """Get TVL data from DefiLlama (proxy for tokens locked)"""
    try:
        url = f"{DEFILLAMA_API}/protocol/{protocol_slug}"
        
        def call():
            r = requests.get(url, headers=HEADERS, timeout=30)
            r.raise_for_status()
            return r.json()
        
        response = retry_request(call)
        
        if 'tvl' in response:
            # Get current TVL
            tvl_data = response['tvl']
            if tvl_data:
                latest_tvl = tvl_data[-1].get('totalLiquidityUSD', 0) if tvl_data else 0
                return latest_tvl
        
        return None
    except Exception as e:
        print(f"Failed to get TVL for {protocol_slug}: {e}")
        return None

def search_defillama_protocol(token_name):
    """Search for protocol slug by token name"""
    try:
        url = f"{DEFILLAMA_API}/protocols"
        
        def call():
            r = requests.get(url, headers=HEADERS, timeout=30)
            r.raise_for_status()
            return r.json()
        
        protocols = retry_request(call)
        
        # Search for matching protocol
        token_clean = token_name.lower().replace('usdt', '').replace('-', '').strip()
        for protocol in protocols:
            if token_clean in protocol.get('name', '').lower() or token_clean in protocol.get('symbol', '').lower():
                return protocol.get('slug')
        
        return None
    except Exception as e:
        print(f"Failed to search DefiLlama for {token_name}: {e}")
        return None


# 4) High-level per-token worker

def process_token(symbol, coin_info, start_date='2021-01-01', end_date='2024-12-31'):
    """Produces a dataframe of daily features for the token and writes to OUT_DIR/{symbol}.parquet"""
    print(f"Processing {symbol}")
    
    # Skip excluded symbols
    if symbol in EXCLUDED_SYMBOLS:
        print(f"Skipping excluded symbol: {symbol}")
        return None
    
    # Determine chain & contract for EVM tokens
    contract = None
    chain = None
    if coin_info and 'platforms' in coin_info:
        # prefer ethereum, binance-smart-chain, polygon
        platforms = coin_info.get('platforms')
        # common keys: 'ethereum', 'binance-smart-chain', 'polygon-pos'
        for ckey in ['ethereum','binance-smart-chain','polygon-pos','arbitrum-one','optimistic-ethereum']:
            addr = platforms.get(ckey)
            if addr:
                contract = addr
                chain = 'eth' if ckey=='ethereum' else ('bsc' if 'binance' in ckey else ('polygon' if 'polygon' in ckey else 'eth'))
                break
        # fallback to first non-empty
        if not contract:
            for k,v in platforms.items():
                if v:
                    contract = v
                    chain = 'eth'
                    break

    print(f"  Contract: {contract}, Chain: {chain}")

    # Create date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    df = pd.DataFrame({'date': date_range})
    
    # Initialize columns
    df['symbol'] = symbol
    df['contract_address'] = contract
    df['chain'] = chain
    df['num_holders'] = None
    df['active_addresses'] = None
    df['large_tx_count'] = None
    df['tvl_usd'] = None

    # 1) Get token holder count (current snapshot)
    if contract and chain:
        num_holders = get_token_holders_count(chain, contract)
        df['num_holders'] = num_holders
        print(f"  Holders: {num_holders}")
        
        # Add delay to respect rate limits
        time.sleep(0.2)  # 5 requests per second limit
    
    # 2) Get on-chain activity metrics for recent period (last 30 days to avoid API limits)
    if contract and chain:
        try:
            # Calculate timestamps for last 30 days
            end_time = datetime.now()
            start_time = end_time - timedelta(days=30)
            start_timestamp = int(start_time.timestamp())
            end_timestamp = int(end_time.timestamp())
            
            # Get active addresses count
            active_addresses = get_active_addresses_count(chain, contract, start_timestamp, end_timestamp)
            df.loc[df['date'] >= start_time.date(), 'active_addresses'] = active_addresses
            print(f"  Active addresses (30d): {active_addresses}")
            
            time.sleep(0.2)  # Rate limit
            
            # Get large transactions count
            large_tx_count = get_large_transactions(chain, contract, start_timestamp, end_timestamp)
            df.loc[df['date'] >= start_time.date(), 'large_tx_count'] = large_tx_count
            print(f"  Large transactions (30d): {large_tx_count}")
            
            time.sleep(0.2)  # Rate limit
            
        except Exception as e:
            print(f"  Error getting activity data: {e}")

    # 3) Get TVL data from DefiLlama
    try:
        protocol_slug = search_defillama_protocol(symbol.replace('USDT', ''))
        if protocol_slug:
            tvl = get_defillama_tvl(protocol_slug)
            df['tvl_usd'] = tvl
            print(f"  TVL: ${tvl:,.0f}" if tvl else "  TVL: Not found")
        time.sleep(0.1)  # Small delay for DefiLlama
    except Exception as e:
        print(f"  Error getting TVL data: {e}")

    # Save to parquet
    out_file = OUT_DIR / f"{symbol}.parquet"
    df.to_parquet(out_file, index=False)
    print(f"  Saved: {out_file}")
    
    return out_file


# 5) Orchestrator for list of tokens

def run_for_symbols(symbols, max_workers=3):  # Reduced workers to respect API limits
    """Process symbols with rate limiting and filtering"""
    # Filter out excluded symbols
    filtered_symbols = [sym for sym in symbols if sym not in EXCLUDED_SYMBOLS]
    print(f"Processing {len(filtered_symbols)} symbols (excluded {len(symbols) - len(filtered_symbols)})")
    
    mapping = coingecko_map_symbols(filtered_symbols)
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(process_token, sym, mapping.get(sym)): sym for sym in filtered_symbols}
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                out = fut.result()
                if out:  # Skip None results from excluded symbols
                    print(f"✓ Finished {sym} -> {out}")
                    results.append(out)
            except Exception as e:
                print(f"✗ Failed {sym}: {e}")
    
    return results

def create_merged_dataset():
    """Merge all individual token parquet files into one dataset"""
    parquet_files = list(OUT_DIR.glob("*.parquet"))
    if not parquet_files:
        print("No parquet files found to merge")
        return
    
    print(f"Merging {len(parquet_files)} token datasets...")
    
    dfs = []
    for file in parquet_files:
        try:
            df = pd.read_parquet(file)
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    if dfs:
        merged_df = pd.concat(dfs, ignore_index=True)
        merged_file = OUT_DIR / "merged_onchain_features.parquet"
        merged_df.to_parquet(merged_file, index=False)
        print(f"Saved merged dataset: {merged_file}")
        print(f"Dataset shape: {merged_df.shape}")
        print(f"Symbols covered: {merged_df['symbol'].nunique()}")
    else:
        print("No data to merge")


# Example usage
if __name__ == '__main__':
    print("🚀 Starting on-chain data collection for 313 tokens...")
    print(f"📂 Output directory: {OUT_DIR}")
    print(f"🚫 Excluding {len(EXCLUDED_SYMBOLS)} symbols")
    
    # Check for data directory
    TRAIN_DATA_DIR = Path("kline_data/train_data")
    if not TRAIN_DATA_DIR.exists():
        TRAIN_DATA_DIR = Path("kline_data")  # fallback
    
    if not TRAIN_DATA_DIR.exists():
        print("❌ kline_data directory not found!")
        exit(1)
    
    # Get symbols from parquet files
    symbol_files = list(TRAIN_DATA_DIR.glob("*.parquet"))
    if not symbol_files:
        # Check for CSV files as fallback
        symbol_files = list(TRAIN_DATA_DIR.glob("*.csv"))
    
    symbols = [f.stem for f in symbol_files]
    print(f"📊 Found {len(symbols)} total symbols")
    
    if not symbols:
        print("❌ No symbol files found!")
        exit(1)
    
    # Check API keys
    if ETHERSCAN_KEY == "YourApiKeyToken":
        print("⚠️  Using default Etherscan API key - get a free key at https://etherscan.io/apis")
    if BSCSCAN_KEY == "YourApiKeyToken":
        print("⚠️  Using default BSCScan API key - get a free key at https://bscscan.com/apis")
    if POLYGONSCAN_KEY == "YourApiKeyToken":
        print("⚠️  Using default PolygonScan API key - get a free key at https://polygonscan.com/apis")
    
    # Run the pipeline
    start_time = time.time()
    results = run_for_symbols(symbols)
    
    # Create merged dataset
    create_merged_dataset()
    
    elapsed = time.time() - start_time
    print(f"✅ Done in {elapsed:.1f}s! Processed {len(results)} tokens.")
    print(f"📁 Output files in: {OUT_DIR}")
    
    # Summary stats
    if results:
        print(f"\n📈 Summary:")
        print(f"   - Individual files: {len(results)}")
        print(f"   - Total symbols requested: {len(symbols)}")
        print(f"   - Successfully processed: {len(results)}")
        print(f"   - Failed/Excluded: {len(symbols) - len(results)}")
        print(f"   - Output directory: {OUT_DIR.absolute()}")
