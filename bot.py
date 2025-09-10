import os
import requests
import pandas as pd
from tqdm import tqdm
import discord
import asyncio
from discord.ext import commands
from dotenv import load_dotenv

load_dotenv()

# -------- API 與 Vegas 通道函數 --------
def get_token_rate(symbol: str):
    host = "https://api.gateio.ws"
    prefix = "/api/v4"
    headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}
    url = '/loan/multi_collateral/current_rate'
    query_param = f'currencies={symbol}'

    try:
        r = requests.get(host + prefix + url + "?" + query_param, headers=headers)
        r.raise_for_status()
        data = r.json()[0]

        if not data:
            return None

        hourly_rate_str = data.get('current_rate')
        if hourly_rate_str is None:
            return None

        hourly_rate = float(hourly_rate_str)
        compound_apr = (1 + hourly_rate) ** 8760 - 1  # 年化利率 (複利)
        return {"symbol": symbol, "hourly_rate": hourly_rate, "compound_apr": compound_apr}

    except:
        return None

def get_all_symbols():
    url = "https://api.gateio.ws/api/v4/spot/currency_pairs"
    try:
        r = requests.get(url)
        r.raise_for_status()
        return [item['id'] for item in r.json()]
    except Exception as e:
        print(f"取得交易對失敗: {e}")
        return []

def get_klines(symbol, interval='1h', limit=700):
    url = f"https://api.gateio.ws/api/v4/spot/candlesticks"
    params = {'currency_pair': symbol, 'interval': interval, 'limit': limit}
    try:
        r = requests.get(url, params=params)
        r.raise_for_status()
        data = r.json()
        df = pd.DataFrame(data)
        df.columns = ['time', 'volume', 'close', 'high', 'low', 'open', 'quote_volume', 'trades']
        for col in ['open', 'high', 'low', 'close']:
            df[col] = df[col].astype(float)
        return df
    except Exception as e:
        print(f"{symbol} K線取得失敗: {e}")
        return None

def detect_vegas_turning_points(df):
    if df is None or len(df) < 676:
        return None

    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema144'] = df['close'].ewm(span=144, adjust=False).mean()
    df['ema169'] = df['close'].ewm(span=169, adjust=False).mean()
    df['ema576'] = df['close'].ewm(span=576, adjust=False).mean()
    df['ema676'] = df['close'].ewm(span=676, adjust=False).mean()

    df['vegas_high'] = df[['ema144', 'ema169']].max(axis=1)
    df['vegas_low'] = df[['ema144', 'ema169']].min(axis=1)
    df['vegas_signal'] = None

    for i in range(1, len(df)):
        prev = df.iloc[i-1]
        curr = df.iloc[i]

        # 多頭條件
        is_breakout = prev['close'] < prev['vegas_low'] and curr['close'] > curr['vegas_high'] and curr['ema12'] > curr['vegas_high']
        is_bounce = prev['close'] > prev['vegas_high'] and curr['close'] > curr['vegas_high'] and min(curr['low'], prev['low']) <= curr['vegas_high'] and curr['ema12'] > curr['vegas_high']
        # 空頭條件
        is_breakdown = prev['close'] > prev['vegas_high'] and curr['close'] < curr['vegas_low'] and curr['ema12'] < curr['vegas_low']
        is_failed_bounce = prev['close'] < prev['vegas_low'] and curr['close'] < curr['vegas_low'] and max(curr['high'], prev['high']) >= curr['vegas_low'] and curr['ema12'] < curr['vegas_low']

        if is_breakout:
            df.at[i, 'vegas_signal'] = 'LONG_BREAKOUT'
        elif is_bounce:
            df.at[i, 'vegas_signal'] = 'LONG_BOUNCE'
        elif is_breakdown:
            df.at[i, 'vegas_signal'] = 'SHORT_BREAKDOWN'
        elif is_failed_bounce:
            df.at[i, 'vegas_signal'] = 'SHORT_FAILED_BOUNCE'

    return df.tail(1)

# -------- Discord Bot 部分 --------
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
CHANNEL_ID = int(os.getenv("CHANNEL_ID"))  # 必須轉 int

intents = discord.Intents.default()
bot = commands.Bot(command_prefix="!", intents=intents)

# 同步收集 Vegas 訊號
def collect_signals():
    symbols = get_all_symbols()
    all_signals = []

    for s in tqdm(symbols[:]):  # 可調整取多少交易對 (太多會很慢)
        df = get_klines(s, interval='1h', limit=700)
        df = detect_vegas_turning_points(df)
        if df is not None:
            signals_df = df[df['vegas_signal'].notna()].copy()
            if not signals_df.empty:
                signals_df['symbol'] = s
                try:
                    rate_info = get_token_rate(s.split('_')[0])
                    signals_df['compound_apr'] = rate_info['compound_apr'] if rate_info else None
                except:
                    signals_df['compound_apr'] = None
                all_signals.append(signals_df)

    if not all_signals:
        return None
    return pd.concat(all_signals, ignore_index=True)

# 非阻塞發送訊號
async def send_vegas_signals():
    loop = asyncio.get_running_loop()
    final_df = await loop.run_in_executor(None, collect_signals)

    channel = bot.get_channel(CHANNEL_ID)

    if final_df is None or final_df.empty:
        await channel.send("目前沒有符合 Vegas 通道轉折條件的交易對。")
        return

    # 分多空，並按 compound_apr 排序
    long_df = final_df[final_df['vegas_signal'].isin(['LONG_BREAKOUT','LONG_BOUNCE'])].sort_values(by='compound_apr', ascending=False).head(5)
    short_df = final_df[final_df['vegas_signal'].isin(['SHORT_BREAKDOWN','SHORT_FAILED_BOUNCE'])].sort_values(by='compound_apr', ascending=False).head(5)

    msg = "**Vegas 通道訊號**\n\n"
    if not long_df.empty:
        msg += "**多頭訊號 (Top 5 by APR)**\n"
        for _, row in long_df.iterrows():
            msg += f"{row['symbol']} | Close: {row['close']} | Signal: {row['vegas_signal']} | APR: {row['compound_apr']:.2%}\n"
        msg += "\n"
    if not short_df.empty:
        msg += "**空頭訊號 (Top 5 by APR)**\n"
        for _, row in short_df.iterrows():
            msg += f"{row['symbol']} | Close: {row['close']} | Signal: {row['vegas_signal']} | APR: {row['compound_apr']:.2%}\n"

    await channel.send(msg)

@bot.event
async def on_ready():
    print(f'已登入 Discord: {bot.user}')
    asyncio.create_task(send_vegas_signals())  # 啟動背景任務

bot.run(DISCORD_TOKEN)
