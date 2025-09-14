import os
import asyncio
import aiohttp
import pandas as pd
from tqdm import tqdm
import discord
from discord.ext import commands
from dotenv import load_dotenv
from aiolimiter import AsyncLimiter

load_dotenv()

# --- 定義每個 API 端點的速率限制器 ---
# Gate.io 的公共端點限制為每 10 秒 200 次請求。
# 我們將限制設為每 10 秒 199 次，以確保不會觸發限速。
# AsyncLimiter(最大請求數, 時間間隔)
all_symbols_limiter = AsyncLimiter(199, 10)
klines_limiter = AsyncLimiter(199, 10)
token_rate_limiter = AsyncLimiter(199, 10)

# Define a semaphore to limit concurrent requests
SEMAPHORE_LIMIT = 10
semaphore = asyncio.Semaphore(SEMAPHORE_LIMIT)

# -------- API 與 Vegas 通道函數 (非同步版本) --------
async def get_all_symbols_async(session):
    async with all_symbols_limiter: # 使用專屬的速率限制器
        url = "https://api.gateio.ws/api/v4/spot/currency_pairs"
        try:
            async with session.get(url) as r:
                r.raise_for_status()
                data = await r.json()
                return [item['id'] for item in data]
        except Exception as e:
            print(f"取得交易對失敗: {e}")
            return []

async def get_klines_async(session, symbol, interval='1h', limit=700, retries=3, backoff_factor=1.0):
    # Use semaphore to control concurrency
    async with semaphore:
        for attempt in range(retries):
            async with klines_limiter: # 使用專屬的速率限制器
                url = "https://api.gateio.ws/api/v4/spot/candlesticks"
                params = {'currency_pair': symbol, 'interval': interval, 'limit': limit}
                try:
                    async with session.get(url, params=params) as r:
                        r.raise_for_status()
                        data = await r.json()
                        if not data:
                            return None
                        df = pd.DataFrame(data)
                        df.columns = ['time', 'volume', 'close', 'high', 'low', 'open', 'quote_volume', 'trades']
                        for col in ['open', 'high', 'low', 'close']:
                            df[col] = df[col].astype(float)
                        return df
                except aiohttp.ClientResponseError as e:
                    if e.status == 429:
                        wait_time = backoff_factor * (2 ** attempt)
                        print(f"取得 {symbol} K線失敗: 429 Too Many Requests. 在 {wait_time:.2f} 秒後重試...")
                        await asyncio.sleep(wait_time)
                    else:
                        print(f"{symbol} K線取得失敗: {e}")
                        return None
                except Exception as e:
                    print(f"{symbol} K線取得失敗: {e}")
                    return None
        print(f"已達 {symbol} K線取得最大重試次數，放棄。")
        return None

async def get_token_rate_async(session, symbol: str):
    # 使用 semaphore 來控制並發數
    async with semaphore:
        async with token_rate_limiter: # 使用專屬的速率限制器
            host = "https://api.gateio.ws"
            prefix = "/api/v4"
            url = '/loan/multi_collateral/current_rate'
            query_param = f'currencies={symbol}'
            try:
                async with session.get(host + prefix + url + "?" + query_param) as r:
                    r.raise_for_status()
                    data = await r.json()
                    if not data:
                        return None
                    
                    hourly_rate_str = data[0].get('current_rate')
                    if hourly_rate_str is None:
                        return None
                    
                    hourly_rate = float(hourly_rate_str)
                    compound_apr = (1 + hourly_rate) ** 8760 - 1
                    return {"symbol": symbol, "hourly_rate": hourly_rate, "compound_apr": compound_apr}
            except Exception as e:
                # print(f"取得 {symbol} 年利率失敗: {e}")
                return None

# 維持 Vegas 通道偵測的同步邏輯
def detect_vegas_turning_points(df):
    if df is None or len(df) < 676:
        return None
    
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema144'] = df['close'].ewm(span=144, adjust=False).mean()
    df['ema169'] = df['close'].ewm(span=169, adjust=False).mean()
    df['vegas_high'] = df[['ema144', 'ema169']].max(axis=1)
    df['vegas_low'] = df[['ema144', 'ema169']].min(axis=1)
    df['vegas_signal'] = None

    for i in range(1, len(df)):
        prev = df.iloc[i-1]
        curr = df.iloc[i]
        
        is_breakout = prev['close'] < prev['vegas_low'] and curr['close'] > curr['vegas_high'] and curr['ema12'] > curr['vegas_high']
        is_bounce = prev['close'] > prev['vegas_high'] and curr['close'] > curr['vegas_high'] and min(curr['low'], prev['low']) <= curr['vegas_high'] and curr['ema12'] > curr['vegas_high']
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
CHANNEL_ID = int(os.getenv("CHANNEL_ID"))

intents = discord.Intents.default()
bot = commands.Bot(command_prefix="!", intents=intents)

async def collect_signals_async():
    async with aiohttp.ClientSession() as session:
        symbols = await get_all_symbols_async(session)
        
        all_signals = []
        # Process symbols one by one to avoid high memory usage
        for symbol in tqdm(symbols[:200], desc="收集並處理 Vegas 訊號"):
            df = await get_klines_async(session, symbol)
            if df is not None:
                signals_df = detect_vegas_turning_points(df)
                if signals_df is not None and not signals_df[signals_df['vegas_signal'].notna()].empty:
                    signals_df = signals_df[signals_df['vegas_signal'].notna()].copy()
                    signals_df['symbol'] = symbol
                    all_signals.append(signals_df)

        if not all_signals:
            return None
        
        final_df = pd.concat(all_signals, ignore_index=True)
        
        apr_tasks = [get_token_rate_async(session, s.split('_')[0]) for s in final_df['symbol'].unique()]
        apr_results = await asyncio.gather(*apr_tasks)
        
        apr_map = {res['symbol']: res['compound_apr'] for res in apr_results if res}
        final_df['compound_apr'] = final_df['symbol'].apply(lambda x: apr_map.get(x.split('_')[0]))

        return final_df

async def send_vegas_signals():
    channel = bot.get_channel(CHANNEL_ID)
    if channel is None:
        print(f"無法找到頻道 ID: {CHANNEL_ID}")
        return

    # 發送分析中的訊息
    analyzing_embed = discord.Embed(
        title="🔍 Vegas 通道分析中...",
        description="正在掃描所有交易對並計算技術指標，請稍候...",
        color=0xFFD700  # 金色
    )
    analyzing_embed.set_footer(text="預計需要 1-2 分鐘完成分析")
    await channel.send(embed=analyzing_embed)

    final_df = await collect_signals_async()

    if final_df is None or final_df.empty:
        no_signals_embed = discord.Embed(
            title="📊 Vegas 通道分析結果",
            description="目前沒有符合 Vegas 通道轉折條件的交易對。",
            color=0x808080  # 灰色
        )
        no_signals_embed.set_footer(text="建議稍後再次檢查")
        await channel.send(embed=no_signals_embed)
        return

    long_df = final_df[final_df['vegas_signal'].isin(['LONG_BREAKOUT', 'LONG_BOUNCE'])].sort_values(by='compound_apr', ascending=False).head(5)
    short_df = final_df[final_df['vegas_signal'].isin(['SHORT_BREAKDOWN', 'SHORT_FAILED_BOUNCE'])].sort_values(by='compound_apr', ascending=False).head(5)

    # 創建主要結果 Embed
    main_embed = discord.Embed(
        title="📈 Vegas 通道訊號分析",
        description="基於 EMA144/169 通道的技術分析結果",
        color=0x00FF00  # 綠色
    )
    
    # 添加統計信息
    total_signals = len(final_df)
    long_count = len(long_df)
    short_count = len(short_df)
    
    main_embed.add_field(
        name="📊 訊號統計",
        value=f"```\n總訊號數: {total_signals}\n多頭訊號: {long_count}\n空頭訊號: {short_count}```",
        inline=False
    )

    # 多頭訊號部分
    if not long_df.empty:
        long_signals = []
        for i, (_, row) in enumerate(long_df.iterrows(), 1):
            signal_emoji = "🚀" if row['vegas_signal'] == 'LONG_BREAKOUT' else "⬆️"
            signal_name = "突破" if row['vegas_signal'] == 'LONG_BREAKOUT' else "反彈"
            apr_str = f"{row['compound_apr']:.2%}" if pd.notna(row['compound_apr']) else "N/A"
            
            long_signals.append(
                f"`{i:2d}.` **{row['symbol']}** {signal_emoji}\n"
                f"     💰 價格: `${row['close']:.6f}`\n"
                f"     📊 類型: `{signal_name}`\n"
                f"     🏦 年利率: `{apr_str}`"
            )
        
        long_text = "\n\n".join(long_signals)
        main_embed.add_field(
            name="🟢 多頭訊號 (前5名 by APR)",
            value=long_text,
            inline=True
        )

    # 空頭訊號部分
    if not short_df.empty:
        short_signals = []
        for i, (_, row) in enumerate(short_df.iterrows(), 1):
            signal_emoji = "📉" if row['vegas_signal'] == 'SHORT_BREAKDOWN' else "⬇️"
            signal_name = "跌破" if row['vegas_signal'] == 'SHORT_BREAKDOWN' else "失敗反彈"
            apr_str = f"{row['compound_apr']:.2%}" if pd.notna(row['compound_apr']) else "N/A"
            
            short_signals.append(
                f"`{i:2d}.` **{row['symbol']}** {signal_emoji}\n"
                f"     💰 價格: `${row['close']:.6f}`\n"
                f"     📊 類型: `{signal_name}`\n"
                f"     🏦 年利率: `{apr_str}`"
            )
        
        short_text = "\n\n".join(short_signals)
        main_embed.add_field(
            name="🔴 空頭訊號 (前5名 by APR)",
            value=short_text,
            inline=True
        )

    # 添加說明和時間戳
    main_embed.add_field(
        name="ℹ️ 使用說明",
        value="```\n🚀 突破: 價格突破 Vegas 通道上方\n⬆️ 反彈: 在通道上方獲得支撐\n📉 跌破: 價格跌破 Vegas 通道下方\n⬇️ 失敗反彈: 反彈失敗轉為空頭```",
        inline=False
    )
    
    main_embed.set_footer(text="⚠️ 僅供參考，請自行評估風險 | Vegas 通道基於 EMA144/169")
    main_embed.timestamp = discord.utils.utcnow()

    await channel.send(embed=main_embed)

@bot.event
async def on_ready():
    print(f'已登入 Discord: {bot.user}')
    # Run the signal collection task
    await send_vegas_signals()
    # Once the task is complete, close the bot gracefully
    await bot.close()

if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)