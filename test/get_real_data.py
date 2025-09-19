import requests
import pandas as pd
import numpy as np
from datetime import datetime
import time

def get_btc_data(symbol='BTC_USDT', interval='6h', total_candles=5000):
    """
    從Gate.io API獲取指定數量的歷史K線數據，通過分頁請求突破單次1000條的限制。
    
    Args:
        symbol (str): 交易對.
        interval (str): 時間間隔 (e.g., '6h', '1d').
        total_candles (int): 想要獲取的總K線數量.
    
    Returns:
        pandas.DataFrame: 包含OHLCV數據的DataFrame.
    """
    print(f"=== 開始獲取 {total_candles} 條 {symbol} {interval} 的歷史數據... ===")
    
    all_data = []
    limit_per_call = 1000 # API單次請求上限
    end_timestamp = None # 從當前時間開始

    while len(all_data) < total_candles:
        remaining = total_candles - len(all_data)
        current_limit = min(remaining, limit_per_call)
        
        url = "https://api.gateio.ws/api/v4/spot/candlesticks"
        params = {
            'currency_pair': symbol,
            'interval': interval,
            'limit': current_limit
        }
        if end_timestamp:
            params['to'] = end_timestamp

        try:
            print(f"發起請求: 獲取 {current_limit} 條數據 (已獲取 {len(all_data)}/{total_candles})...")
            response = requests.get(url, params=params, timeout=20)
            response.raise_for_status()
            
            chunk = response.json()
            if not chunk:
                print("API返回空數據，已到達最早的歷史記錄。")
                break
            
            all_data.extend(chunk)
            
            oldest_timestamp_in_chunk = int(chunk[-1][0])
            end_timestamp = oldest_timestamp_in_chunk - 1

            time.sleep(1)

        except requests.exceptions.RequestException as e:
            print(f"❌ API請求失敗: {e}")
            print("將使用備用模擬數據...")
            return create_fallback_data(total_candles)
        except Exception as e:
            print(f"❌ 數據處理錯誤: {e}")
            print("將使用備用模擬數據...")
            return create_fallback_data(total_candles)

    if not all_data:
        print("❌ 未能獲取任何數據，將使用備用模擬數據。")
        return create_fallback_data(total_candles)

    df_data = []
    for candle in all_data:
        df_data.append({
            'time': pd.to_datetime(int(candle[0]), unit='s'),
            'volume': float(candle[1]),
            'close': float(candle[2]),
            'high': float(candle[3]),
            'low': float(candle[4]),
            'open': float(candle[5])
        })
    
    df = pd.DataFrame(df_data)
    df = df.drop_duplicates(subset='time').sort_values('time').reset_index(drop=True)
    
    print(f"\n✅ 成功獲取並整理了 {len(df)} 條數據。")
    if not df.empty:
        print(f"時間範圍: {df['time'].iloc[0]} 到 {df['time'].iloc[-1]}")
        print(f"價格範圍: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
    
    return df

def create_fallback_data(n=200):
    """
    創建備用的模擬數據（當API不可用時）
    """
    print("正在創建備用模擬數據...")
    
    np.random.seed(42)
    price_base = 45000
    returns = np.random.normal(0.0005, 0.02, n)
    price_changes = np.cumprod(1 + returns)
    closes = price_base * price_changes
    
    opens = np.zeros(n)
    highs = np.zeros(n)
    lows = np.zeros(n)
    
    opens[0] = price_base
    for i in range(1, n):
        opens[i] = closes[i-1] * np.random.normal(1, 0.001)
    
    for i in range(n):
        daily_range = abs(closes[i] - opens[i]) + closes[i] * np.random.uniform(0.005, 0.015)
        
        if closes[i] > opens[i]:
            highs[i] = closes[i] + np.random.uniform(0.1, 0.4) * daily_range
            lows[i] = opens[i] - np.random.uniform(0.1, 0.3) * daily_range
        else:
            highs[i] = opens[i] + np.random.uniform(0.1, 0.3) * daily_range
            lows[i] = closes[i] - np.random.uniform(0.1, 0.4) * daily_range
    
    for i in range(n):
        highs[i] = max(highs[i], opens[i], closes[i])
        lows[i] = min(lows[i], opens[i], closes[i])
    
    volumes = np.random.lognormal(12, 0.4, n)
    
    df = pd.DataFrame({
        'time': pd.date_range(start='2024-01-01', periods=n, freq='1H'),
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    })
    
    print(f"✅ 創建了 {len(df)} 條模擬數據")
    return df

if __name__ == "__main__":
    df = get_btc_data('BTC_USDT', '1h', 2500)
    if df is not None:
        print("\n數據樣本:")
        print(df.head())
        print(df.tail())
        print("\n數據統計:")
        print(f"數據條數: {len(df)}")
