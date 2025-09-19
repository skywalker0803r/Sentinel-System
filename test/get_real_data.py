import requests
import pandas as pd
import numpy as np
from datetime import datetime
import time

def get_btc_data(symbol='BTC_USDT', interval='1h', limit=200):
    """
    從Gate.io API獲取真實的比特幣數據
    
    Args:
        symbol: 交易對 (默認: BTC_USDT)
        interval: 時間間隔 (1m, 5m, 15m, 30m, 1h, 4h, 1d)
        limit: 數據條數 (最大1000)
    
    Returns:
        pandas.DataFrame: 包含OHLCV數據的DataFrame
    """
    
    url = "https://api.gateio.ws/api/v4/spot/candlesticks"
    params = {
        'currency_pair': symbol,
        'interval': interval,
        'limit': limit
    }
    
    try:
        print(f"正在從Gate.io獲取 {symbol} {interval} 數據...")
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if not data:
            raise ValueError("API返回空數據")
        
        # 將數據轉換為DataFrame
        # Gate.io API返回格式: [timestamp, volume, close, high, low, open]
        df_data = []
        for candle in reversed(data):  # 反轉以獲得時間順序
            timestamp = int(candle[0])
            volume = float(candle[1])
            close = float(candle[2])
            high = float(candle[3])
            low = float(candle[4])
            open = float(candle[5])
            
            df_data.append({
                'time': pd.to_datetime(timestamp, unit='s'),
                'open': open,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
        
        df = pd.DataFrame(df_data)
        
        # 確保數據按時間排序
        df = df.sort_values('time').reset_index(drop=True)
        
        print(f"✅ 成功獲取 {len(df)} 條數據")
        print(f"時間範圍: {df['time'].iloc[0]} 到 {df['time'].iloc[-1]}")
        print(f"價格範圍: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
        
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"❌ API請求失敗: {e}")
        print("使用備用模擬數據...")
        return create_fallback_data(limit)
    
    except Exception as e:
        print(f"❌ 數據處理錯誤: {e}")
        print("使用備用模擬數據...")
        return create_fallback_data(limit)

def create_fallback_data(n=200):
    """
    創建備用的模擬數據（當API不可用時）
    """
    print("正在創建備用模擬數據...")
    
    np.random.seed(42)
    price_base = 45000  # BTC基準價格
    
    # 生成真實的價格走勢
    returns = np.random.normal(0.0005, 0.02, n)  # 小幅上漲趨勢
    price_changes = np.cumprod(1 + returns)
    closes = price_base * price_changes
    
    # 生成OHLC
    opens = np.zeros(n)
    highs = np.zeros(n)
    lows = np.zeros(n)
    
    opens[0] = price_base
    for i in range(1, n):
        opens[i] = closes[i-1] * np.random.normal(1, 0.001)
    
    for i in range(n):
        daily_range = abs(closes[i] - opens[i]) + closes[i] * np.random.uniform(0.005, 0.015)
        
        if closes[i] > opens[i]:  # 陽線
            highs[i] = closes[i] + np.random.uniform(0.1, 0.4) * daily_range
            lows[i] = opens[i] - np.random.uniform(0.1, 0.3) * daily_range
        else:  # 陰線
            highs[i] = opens[i] + np.random.uniform(0.1, 0.3) * daily_range
            lows[i] = closes[i] - np.random.uniform(0.1, 0.4) * daily_range
    
    # 確保OHLC邏輯正確
    for i in range(n):
        highs[i] = max(highs[i], opens[i], closes[i])
        lows[i] = min(lows[i], opens[i], closes[i])
    
    # 生成成交量
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
    # 測試數據獲取
    df = get_btc_data('BTC_USDT', '1h', 100)
    print("\n數據樣本:")
    print(df.head())
    print("\n數據統計:")
    print(f"數據條數: {len(df)}")
    print(f"價格範圍: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
    print(f"平均波動: {((df['high'] - df['low']) / df['close'] * 100).mean():.2f}%")