import os
import asyncio
import aiohttp
import pandas as pd
from tqdm import tqdm
import discord
from discord.ext import commands
from dotenv import load_dotenv
from aiolimiter import AsyncLimiter
from smc_indicators import SmartMoneyConceptsAnalyzer

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

# 初始化 Smart Money Concepts 分析器
smc_analyzer = SmartMoneyConceptsAnalyzer()

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
                        # 安全轉換數值列，處理空值和非數值字符串
                        for col in ['open', 'high', 'low', 'close', 'volume']:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        
                        # 移除任何包含 NaN 的行
                        df = df.dropna(subset=['open', 'high', 'low', 'close'])
                        
                        # 如果數據太少，返回 None
                        if len(df) < 100:
                            return None
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

def calculate_signal_score(vegas_signal, smc_analysis, symbol, apr_data):
    """計算綜合訊號評分 (0-100)"""
    score = 0
    factors = {}
    
    # Vegas 通道基礎分數 (0-40分)
    if vegas_signal is not None:
        # 處理不同的輸入類型
        if hasattr(vegas_signal, 'iloc'):  # pandas Series/DataFrame
            signal_type = vegas_signal.get('vegas_signal', None)
            if signal_type is None and hasattr(vegas_signal, 'iloc'):
                signal_type = vegas_signal.iloc[0] if len(vegas_signal) > 0 else None
        elif isinstance(vegas_signal, dict):  # 字典
            signal_type = vegas_signal.get('vegas_signal', None)
        else:  # 直接傳入字符串
            signal_type = vegas_signal
            
        if signal_type in ['LONG_BREAKOUT', 'SHORT_BREAKDOWN']:
            score += 25  # 突破訊號較強
            factors['vegas_breakout'] = 25
        elif signal_type in ['LONG_BOUNCE', 'SHORT_FAILED_BOUNCE']:
            score += 15  # 反彈訊號較弱
            factors['vegas_bounce'] = 15
    
    # SMC 市場結構分數 (0-25分)
    if smc_analysis and 'market_structure' in smc_analysis:
        structure = smc_analysis['market_structure']
        
        # BOS/CHoCH 確認
        if structure['bos_signals']:
            score += 15
            factors['smc_bos'] = 15
        if structure['choch_signals']:
            score += 20  # 趨勢轉變更重要
            factors['smc_choch'] = 20
    
    # Order Blocks 分數 (0-15分)
    if smc_analysis and 'order_blocks' in smc_analysis:
        active_obs = [ob for ob in smc_analysis['order_blocks'] if ob['active']]
        if active_obs:
            ob_score = min(15, len(active_obs) * 3)
            score += ob_score
            factors['order_blocks'] = ob_score
    
    # Fair Value Gaps 分數 (0-10分)
    if smc_analysis and 'fair_value_gaps' in smc_analysis:
        fvgs = smc_analysis['fair_value_gaps']
        if fvgs:
            fvg_score = min(10, len(fvgs) * 2)
            score += fvg_score
            factors['fair_value_gaps'] = fvg_score
    
    # 流動性掃蕩分數 (0-10分)
    if smc_analysis and 'liquidity_sweeps' in smc_analysis:
        sweeps = smc_analysis['liquidity_sweeps']
        if sweeps:
            sweep_score = min(10, len(sweeps) * 2)
            score += sweep_score
            factors['liquidity_sweeps'] = sweep_score
    
    # 年利率加成 (0-10分)
    if apr_data and apr_data > 0:
        if apr_data > 1.0:  # 100%+
            score += 10
            factors['high_apr'] = 10
        elif apr_data > 0.5:  # 50%+
            score += 6
            factors['medium_apr'] = 6
        elif apr_data > 0.2:  # 20%+
            score += 3
            factors['low_apr'] = 3
    
    return min(100, score), factors

def enhance_vegas_with_smc(df, symbol):
    """使用 SMC 增強 Vegas 通道分析"""
    if df is None or len(df) < 676:
        return None
    
    # 獲取 Vegas 訊號
    vegas_df = detect_vegas_turning_points(df)
    
    # 獲取 SMC 分析
    try:
        smc_analysis = smc_analyzer.get_comprehensive_analysis(df)
    except Exception as e:
        print(f"SMC 分析失敗 {symbol}: {e}")
        smc_analysis = {}
    
    if vegas_df is None or vegas_df.empty:
        # 即使沒有 Vegas 訊號，也檢查是否有強 SMC 訊號
        if smc_analysis:
            structure = smc_analysis.get('market_structure', {})
            if (structure.get('bos_signals') or structure.get('choch_signals') or
                smc_analysis.get('order_blocks') or smc_analysis.get('liquidity_sweeps')):
                
                # 創建純 SMC 訊號
                current_price = df['close'].iloc[-1]
                smc_signal_type = 'SMC_BULLISH' if smc_analysis.get('overall_bias') == 'BULLISH' else 'SMC_BEARISH'
                
                result_df = pd.DataFrame({
                    'close': [current_price],
                    'vegas_signal': [smc_signal_type],
                    'ema12': [df['close'].ewm(span=12).mean().iloc[-1]],
                    'vegas_high': [0],
                    'vegas_low': [0]
                })
                result_df['smc_analysis'] = [smc_analysis]
                result_df['signal_source'] = ['SMC_ONLY']
                return result_df
        return None
    
    # 如果有 Vegas 訊號，加入 SMC 分析
    vegas_df = vegas_df.copy()
    vegas_df['smc_analysis'] = [smc_analysis]
    vegas_df['signal_source'] = ['VEGAS_SMC']
    
    return vegas_df

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
        for symbol in tqdm(symbols[:], desc="收集並處理 Vegas + SMC 訊號"):
            try:
                df = await get_klines_async(session, symbol)
                if df is not None and len(df) >= 676:  # 確保有足夠的數據
                    # 使用增強的分析函數
                    enhanced_signals = enhance_vegas_with_smc(df, symbol)
                    if enhanced_signals is not None and not enhanced_signals[enhanced_signals['vegas_signal'].notna()].empty:
                        enhanced_signals = enhanced_signals[enhanced_signals['vegas_signal'].notna()].copy()
                        enhanced_signals['symbol'] = symbol
                        all_signals.append(enhanced_signals)
            except Exception as e:
                print(f"處理 {symbol} 時發生錯誤: {e}")
                continue

        if not all_signals:
            return None
        
        final_df = pd.concat(all_signals, ignore_index=True)
        
        apr_tasks = [get_token_rate_async(session, s.split('_')[0]) for s in final_df['symbol'].unique()]
        apr_results = await asyncio.gather(*apr_tasks)
        
        apr_map = {res['symbol']: res['compound_apr'] for res in apr_results if res}
        final_df['compound_apr'] = final_df['symbol'].apply(lambda x: apr_map.get(x.split('_')[0]))
        
        # 計算綜合評分
        scores = []
        score_factors = []
        smc_data = []
        
        for idx, row in final_df.iterrows():
            smc_analysis = row.get('smc_analysis', {})
            apr_data = row.get('compound_apr', 0)
            
            score, factors = calculate_signal_score(row, smc_analysis, row['symbol'], apr_data)
            scores.append(score)
            score_factors.append(factors)
            smc_data.append(smc_analysis)
        
        final_df['signal_score'] = scores
        final_df['score_factors'] = score_factors
        final_df['smc_data'] = smc_data

        return final_df

async def send_enhanced_signals():
    channel = bot.get_channel(CHANNEL_ID)
    if channel is None:
        print(f"無法找到頻道 ID: {CHANNEL_ID}")
        return

    # 發送分析中的訊息
    analyzing_embed = discord.Embed(
        title="🔍 增強版技術分析中...",
        description="正在掃描所有交易對並計算 Vegas 通道 + Smart Money Concepts 指標，請稍候...",
        color=0xFFD700  # 金色
    )
    analyzing_embed.set_footer(text="預計需要 2-3 分鐘完成分析")
    await channel.send(embed=analyzing_embed)

    final_df = await collect_signals_async()

    if final_df is None or final_df.empty:
        no_signals_embed = discord.Embed(
            title="📊 技術分析結果",
            description="目前沒有符合條件的交易訊號。",
            color=0x808080  # 灰色
        )
        no_signals_embed.set_footer(text="建議稍後再次檢查")
        await channel.send(embed=no_signals_embed)
        return

    # 按評分分層處理訊號
    final_df = final_df.sort_values(by='signal_score', ascending=False)
    
    # Tier 1: 高信心訊號 (評分 >= 70)
    tier1_df = final_df[final_df['signal_score'] >= 70].head(3)
    
    # Tier 2: 中信心訊號 (評分 50-69)
    tier2_df = final_df[(final_df['signal_score'] >= 50) & (final_df['signal_score'] < 70)].head(5)
    
    # Tier 3: 觀察訊號 (評分 30-49)
    tier3_df = final_df[(final_df['signal_score'] >= 30) & (final_df['signal_score'] < 50)].head(5)

    # 統計數據
    total_signals = len(final_df)
    vegas_signals = len(final_df[final_df.get('signal_source', 'VEGAS') == 'VEGAS_SMC'])
    smc_only_signals = len(final_df[final_df.get('signal_source', 'VEGAS') == 'SMC_ONLY'])
    avg_score = final_df['signal_score'].mean() if not final_df.empty else 0

    # 創建主要結果 Embed
    main_embed = discord.Embed(
        title="🎯 增強版技術分析報告",
        description="結合 Vegas 通道 + Smart Money Concepts 的綜合分析",
        color=0x00FF00  # 綠色
    )
    
    # 添加統計信息
    main_embed.add_field(
        name="📊 分析統計",
        value=f"```\n總訊號數: {total_signals}\nVegas+SMC: {vegas_signals}\n純SMC訊號: {smc_only_signals}\n平均評分: {avg_score:.1f}/100```",
        inline=False
    )

    # Tier 1: 高信心訊號
    if not tier1_df.empty:
        tier1_signals = []
        for i, (_, row) in enumerate(tier1_df.iterrows(), 1):
            signal_type = row['vegas_signal']
            signal_emoji = get_signal_emoji(signal_type)
            signal_name = get_signal_name(signal_type)
            apr_str = f"{row['compound_apr']:.2%}" if pd.notna(row['compound_apr']) else "N/A"
            score = row['signal_score']
            
            # 獲取 SMC 亮點
            smc_highlights = get_smc_highlights(row.get('smc_data', {}))
            
            # 獲取評分明細
            score_breakdown = format_score_breakdown(row.get('score_factors', {}))
            
            tier1_signals.append(
                f"`{i}.` **{row['symbol']}** {signal_emoji} `{score:.0f}分`\n"
                f"     💰 `${row['close']:.6f}` | 📊 `{signal_name}` | 🏦 `{apr_str}`\n"
                f"     🎯 {smc_highlights}\n"
                f"     📊 **評分明細**: {score_breakdown}"
            )
        
        tier1_text = "\n\n".join(tier1_signals)
        main_embed.add_field(
            name="🥇 Tier 1: 高信心訊號 (70-100分)",
            value=tier1_text,
            inline=False
        )

    # Tier 2: 中信心訊號
    if not tier2_df.empty:
        tier2_signals = []
        for i, (_, row) in enumerate(tier2_df.iterrows(), 1):
            signal_type = row['vegas_signal']
            signal_emoji = get_signal_emoji(signal_type)
            apr_str = f"{row['compound_apr']:.2%}" if pd.notna(row['compound_apr']) else "N/A"
            score = row['signal_score']
            
            # 簡化的評分明細
            score_breakdown = format_score_breakdown(row.get('score_factors', {}))
            
            tier2_signals.append(
                f"`{i}.` **{row['symbol']}** {signal_emoji} `{score:.0f}分` | `{apr_str}`\n"
                f"     📊 {score_breakdown}"
            )
        
        tier2_text = "\n\n".join(tier2_signals)
        main_embed.add_field(
            name="🥈 Tier 2: 中信心訊號 (50-69分)",
            value=tier2_text,
            inline=True
        )

    # Tier 3: 觀察訊號
    if not tier3_df.empty:
        tier3_signals = []
        for i, (_, row) in enumerate(tier3_df.iterrows(), 1):
            signal_type = row['vegas_signal']
            signal_emoji = get_signal_emoji(signal_type)
            score = row['signal_score']
            
            # 簡化的評分明細
            score_breakdown = format_score_breakdown(row.get('score_factors', {}))
            
            tier3_signals.append(
                f"`{i}.` **{row['symbol']}** {signal_emoji} `{score:.0f}分`\n"
                f"     📊 {score_breakdown}"
            )
        
        tier3_text = "\n\n".join(tier3_signals)
        main_embed.add_field(
            name="🥉 Tier 3: 觀察清單 (30-49分)",
            value=tier3_text,
            inline=True
        )

    # 添加說明
    main_embed.add_field(
        name="ℹ️ 評分說明",
        value="```\n📊 Vegas通道: 25分 (突破) / 15分 (反彈)\n🏗️ SMC結構: 15分 (BOS) / 20分 (CHoCH)\n📦 OrderBlock: 最高15分\n⚡ 流動性掃蕩: 最高10分\n💎 FairValueGap: 最高10分\n💰 高年利率: 最高10分```",
        inline=False
    )
    
    main_embed.set_footer(text="⚠️ 僅供參考，請自行評估風險 | 結合多重技術指標分析")
    main_embed.timestamp = discord.utils.utcnow()

    await channel.send(embed=main_embed)

def get_signal_emoji(signal_type):
    """獲取訊號對應的 emoji"""
    emoji_map = {
        'LONG_BREAKOUT': '🚀',
        'LONG_BOUNCE': '⬆️',
        'SHORT_BREAKDOWN': '📉',
        'SHORT_FAILED_BOUNCE': '⬇️',
        'SMC_BULLISH': '🔥',
        'SMC_BEARISH': '❄️'
    }
    return emoji_map.get(signal_type, '❓')

def get_signal_name(signal_type):
    """獲取訊號名稱"""
    name_map = {
        'LONG_BREAKOUT': '向上突破',
        'LONG_BOUNCE': '向上反彈',
        'SHORT_BREAKDOWN': '向下跌破',
        'SHORT_FAILED_BOUNCE': '失敗反彈',
        'SMC_BULLISH': 'SMC看漲',
        'SMC_BEARISH': 'SMC看跌'
    }
    return name_map.get(signal_type, '未知訊號')

def get_smc_highlights(smc_data):
    """獲取 SMC 分析亮點"""
    if not smc_data:
        return "基礎分析"
    
    highlights = []
    
    # 檢查市場結構
    structure = smc_data.get('market_structure', {})
    if structure.get('bos_signals'):
        highlights.append('BOS確認')
    if structure.get('choch_signals'):
        highlights.append('CHoCH轉勢')
    
    # 檢查 Order Blocks
    order_blocks = smc_data.get('order_blocks', [])
    if order_blocks:
        active_obs = [ob for ob in order_blocks if ob.get('active', False)]
        if active_obs:
            highlights.append(f'{len(active_obs)}個活躍OB')
    
    # 檢查 Fair Value Gaps
    fvgs = smc_data.get('fair_value_gaps', [])
    if fvgs:
        highlights.append(f'{len(fvgs)}個FVG')
    
    # 檢查流動性掃蕩
    sweeps = smc_data.get('liquidity_sweeps', [])
    if sweeps:
        highlights.append('流動性掃蕩')
    
    # 檢查當前區域
    zones = smc_data.get('premium_discount', {})
    current_zone = zones.get('current_zone', '')
    if current_zone:
        zone_emoji = {'PREMIUM': '🔴', 'DISCOUNT': '🟢', 'EQUILIBRIUM': '🟡'}.get(current_zone, '')
        highlights.append(f'{zone_emoji}{current_zone.lower()}')
    
    return ' | '.join(highlights) if highlights else '基礎分析'

def format_score_breakdown(score_factors):
    """格式化評分明細"""
    if not score_factors:
        return "無評分資料"
    
    breakdown_parts = []
    
    # Vegas 通道評分
    if 'vegas_breakout' in score_factors:
        breakdown_parts.append(f"Vegas突破: {score_factors['vegas_breakout']}分")
    elif 'vegas_bounce' in score_factors:
        breakdown_parts.append(f"Vegas反彈: {score_factors['vegas_bounce']}分")
    
    # SMC 結構評分
    if 'smc_bos' in score_factors:
        breakdown_parts.append(f"BOS: {score_factors['smc_bos']}分")
    if 'smc_choch' in score_factors:
        breakdown_parts.append(f"CHoCH: {score_factors['smc_choch']}分")
    
    # Order Blocks 評分
    if 'order_blocks' in score_factors:
        breakdown_parts.append(f"OB: {score_factors['order_blocks']}分")
    
    # Fair Value Gaps 評分
    if 'fair_value_gaps' in score_factors:
        breakdown_parts.append(f"FVG: {score_factors['fair_value_gaps']}分")
    
    # 流動性掃蕩評分
    if 'liquidity_sweeps' in score_factors:
        breakdown_parts.append(f"流動性: {score_factors['liquidity_sweeps']}分")
    
    # APR 評分
    if 'high_apr' in score_factors:
        breakdown_parts.append(f"高APR: {score_factors['high_apr']}分")
    elif 'medium_apr' in score_factors:
        breakdown_parts.append(f"中APR: {score_factors['medium_apr']}分")
    elif 'low_apr' in score_factors:
        breakdown_parts.append(f"低APR: {score_factors['low_apr']}分")
    
    return " | ".join(breakdown_parts) if breakdown_parts else "基礎評分"

@bot.event
async def on_ready():
    print(f'已登入 Discord: {bot.user}')
    # Run the signal collection task
    await send_enhanced_signals()
    # Once the task is complete, close the bot gracefully
    await bot.close()

if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)