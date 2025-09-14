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
async def get_promising_symbols_by_apy(session, top_percentile=0.5, max_symbols=1000):
    """
    基於APY排名篩選有希望的交易對（取前20%）
    
    Args:
        session: aiohttp session
        top_percentile: 取前百分比（預設0.2 = 前20%）
        max_symbols: 最大返回數量
        
    Returns:
        tuple: (symbols_list, apy_dict) - 交易對列表和APY字典
    """
    print("🔍 正在基於APY排名篩選有潛力的交易對（前20%）...")
    
    # 先獲取所有USDT交易對
    async with all_symbols_limiter:
        url = "https://api.gateio.ws/api/v4/spot/currency_pairs"
        try:
            async with session.get(url) as r:
                r.raise_for_status()
                data = await r.json()
                all_symbols = [item['id'] for item in data if item['id'].endswith('_USDT')]
        except Exception as e:
            print(f"取得交易對失敗: {e}")
            return [], {}
    
    # 獲取所有幣種的APY
    print(f"📊 檢查 {len(all_symbols)} 個USDT交易對的APY...")
    promising_symbols = []
    apy_data = []
    all_apy_dict = {}  # 儲存所有APY數據
    
    # 批量獲取APY (每次處理50個以控制速度)
    batch_size = 50
    all_valid_apy_data = []  # 儲存所有有效的APY數據
    for i in range(0, len(all_symbols), batch_size):
        batch_symbols = all_symbols[i:i+batch_size]
        batch_base_coins = [s.split('_')[0] for s in batch_symbols]
        
        # 並行獲取這批幣種的APY
        apr_tasks = [get_token_rate_async(session, coin) for coin in batch_base_coins]
        apr_results = await asyncio.gather(*apr_tasks, return_exceptions=True)
        
        for symbol, apr_result in zip(batch_symbols, apr_results):
            if isinstance(apr_result, dict) and apr_result.get('compound_apr') is not None:
                base_coin = symbol.split('_')[0]
                apy = apr_result['compound_apr']
                all_apy_dict[base_coin] = apy
                
                # 收集所有有效的APY數據（不再用固定門檻）
                all_valid_apy_data.append({
                    'symbol': symbol,
                    'apy': apy
                })
        
        print(f"   ✅ 已檢查 {min(i+batch_size, len(all_symbols))}/{len(all_symbols)} 個交易對")
    
    # 按APY排序
    all_valid_apy_data.sort(key=lambda x: x['apy'], reverse=True)
    
    # 計算前20%的數量
    total_valid_apys = len(all_valid_apy_data)
    top_20_percent_count = int(total_valid_apys * top_percentile)  # 前20%
    
    if top_20_percent_count == 0 and total_valid_apys > 0:
        top_20_percent_count = min(10, total_valid_apys)  # 至少取10個
    
    # 取前20%的高APY幣種
    apy_data = all_valid_apy_data[:min(top_20_percent_count, max_symbols)]
    promising_symbols = [item['symbol'] for item in apy_data]
    
    # 如果結果太少，補充一些主流幣種
    if len(promising_symbols) < 20:
        mainstream_symbols = []
        mainstream_coins = ['BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'SOL', 'DOT', 'MATIC', 'LINK', 'UNI']
        for coin in mainstream_coins:
            symbol = f"{coin}_USDT"
            if symbol in all_symbols and symbol not in promising_symbols:
                mainstream_symbols.append(symbol)
        
        # 補充主流幣種，但不超過max_symbols
        supplement_count = min(len(mainstream_symbols), max_symbols - len(promising_symbols))
        promising_symbols.extend(mainstream_symbols[:supplement_count])
        print(f"💡 補充了 {supplement_count} 個主流幣種")
    
    print(f"🎯 APY排名篩選結果:")
    print(f"   📊 有效APY數據: {total_valid_apys} 個")
    print(f"   🏆 前20%數量: {top_20_percent_count} 個")
    print(f"   ✅ 最終選定: {len(promising_symbols)} 個交易對")
    
    if apy_data:
        print(f"   📈 最高APY: {apy_data[0]['apy']:.2%} ({apy_data[0]['symbol']})")
        if len(apy_data) > 1:
            print(f"   📊 前20%平均APY: {sum(item['apy'] for item in apy_data)/len(apy_data):.2%}")
            print(f"   📉 前20%門檻APY: {apy_data[-1]['apy']:.2%}")
    
    return promising_symbols, all_apy_dict

async def get_all_symbols_async(session, filter_promising=False):
    """
    獲取交易對列表
    
    Args:
        session: aiohttp session
        filter_promising: 是否篩選有希望的交易對
        
    Returns:
        tuple: (symbols_list, apy_dict) - 交易對列表和APY字典（如果有的話）
    """
    if filter_promising:
        # 使用基於APY的智能篩選
        return await get_promising_symbols_by_apy(session)
    else:
        # 返回所有交易對
        async with all_symbols_limiter:
            url = "https://api.gateio.ws/api/v4/spot/currency_pairs"
            try:
                async with session.get(url) as r:
                    r.raise_for_status()
                    data = await r.json()
                    all_symbols = [item['id'] for item in data]
                    print(f"📊 獲取全部交易對: {len(all_symbols)} 個")
                    return all_symbols, {}
            except Exception as e:
                print(f"取得交易對失敗: {e}")
                return [], {}

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

async def collect_signals_async(filter_promising=True):
    """
    收集交易訊號
    
    Args:
        filter_promising: 是否只分析有希望的交易對（預設True）
    """
    async with aiohttp.ClientSession() as session:
        symbols, apy_dict = await get_all_symbols_async(session, filter_promising)
        
        all_signals = []
        # Process symbols one by one to avoid high memory usage
        desc = "分析高APY潛力幣種的技術訊號" if filter_promising else "收集並處理 Vegas + SMC 訊號"
        for symbol in tqdm(symbols[:], desc=desc):
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
        
        # 使用已經獲取的APY數據，避免重複計算
        if filter_promising and apy_dict:
            print("💡 使用已計算的APY數據，避免重複API調用")
            final_df['compound_apr'] = final_df['symbol'].apply(lambda x: apy_dict.get(x.split('_')[0]))
        else:
            # 獲取APY數據
            unique_symbols = final_df['symbol'].unique()
            unique_base_coins = [s.split('_')[0] for s in unique_symbols]
            
            apr_tasks = [get_token_rate_async(session, coin) for coin in unique_base_coins]
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

async def send_enhanced_signals(filter_promising=True):
    """
    發送增強版訊號分析
    
    Args:
        filter_promising: 是否只分析有希望的交易對（預設True）
    """
    channel = bot.get_channel(CHANNEL_ID)
    if channel is None:
        print(f"無法找到頻道 ID: {CHANNEL_ID}")
        return

    # 發送初始分析中的訊息
    if filter_promising:
        analyzing_embed = discord.Embed(
            title="🔍 APY排名篩選分析中...",
            description="正在篩選APY前20%的幣種，這些往往是價格低點的潛力標的...",
            color=0xFFD700  # 金色
        )
        analyzing_embed.set_footer(text="正在進行APY排名篩選和技術分析，請稍候...")
    else:
        analyzing_embed = discord.Embed(
            title="🔍 全面技術分析中...",
            description="正在掃描所有交易對並計算 Vegas 通道 + Smart Money Concepts 指標，請稍候...",
            color=0xFFD700  # 金色
        )
        analyzing_embed.set_footer(text="正在進行全市場掃描，請稍候...")
    
    await channel.send(embed=analyzing_embed)

    final_df = await collect_signals_async(filter_promising)

    if final_df is None or final_df.empty:
        no_signals_embed = discord.Embed(
            title="📊 技術分析結果",
            description="目前沒有符合條件的交易訊號。",
            color=0x808080  # 灰色
        )
        no_signals_embed.set_footer(text="建議稍後再次檢查")
        await channel.send(embed=no_signals_embed)
        return

    # 只篩選做多訊號
    long_signals = ['LONG_BREAKOUT', 'LONG_BOUNCE', 'SMC_BULLISH']
    final_df = final_df[final_df['vegas_signal'].isin(long_signals)]
    
    # 按評分排序，取TOP 10
    final_df = final_df.sort_values(by='signal_score', ascending=False).head(10)

    # 統計數據
    total_signals = len(final_df)
    vegas_signals = len(final_df[final_df.get('signal_source', 'VEGAS') == 'VEGAS_SMC'])
    smc_only_signals = len(final_df[final_df.get('signal_source', 'VEGAS') == 'SMC_ONLY'])
    avg_score = final_df['signal_score'].mean() if not final_df.empty else 0

    # 創建主要結果 Embed
    main_embed = discord.Embed(
        title="🚀 TOP 10 做多訊號分析",
        description="專注做多機會 - Vegas 通道 + Smart Money Concepts",
        color=0x00FF00  # 綠色
    )
    
    # 添加統計信息
    main_embed.add_field(
        name="📊 做多訊號統計",
        value=f"```\n做多訊號數: {total_signals}\nVegas+SMC: {vegas_signals}\n純SMC訊號: {smc_only_signals}\n平均評分: {avg_score:.1f}/100```",
        inline=False
    )

    # TOP 10 做多訊號
    if not final_df.empty:
        top_signals = []
        for i, (_, row) in enumerate(final_df.iterrows(), 1):
            signal_type = row['vegas_signal']
            signal_emoji = get_signal_emoji(signal_type)
            signal_name = get_signal_name(signal_type)
            apr_str = f"{row['compound_apr']:.2%}" if pd.notna(row['compound_apr']) else "N/A"
            score = row['signal_score']
            
            # 獲取 SMC 亮點和價格區間
            smc_highlights, zone_info = get_smc_highlights(row.get('smc_data', {}))
            
            # 獲取評分明細
            score_breakdown = format_score_breakdown(row.get('score_factors', {}))
            
            # 根據排名添加獎牌emoji
            rank_emoji = ""
            if i == 1:
                rank_emoji = "🥇 "
            elif i == 2:
                rank_emoji = "🥈 "
            elif i == 3:
                rank_emoji = "🥉 "
            
            # 格式化價格區間信息（簡化格式以節省字符）
            zone_display = ""
            if zone_info:
                zone_display = f"\n     🔴`{zone_info['high_price_zone']}` 🟢`{zone_info['low_price_zone']}`"
            
            top_signals.append(
                f"{rank_emoji}`{i}.` **{row['symbol']}** {signal_emoji} `{score:.0f}分`\n"
                f"     💰 `${row['close']:.6f}` | 📊 `{signal_name}` | 🏦 `{apr_str}`\n"
                f"     🎯 {smc_highlights}{zone_display}\n"
                f"     📊 **評分明細**: {score_breakdown}"
            )
        
        # 將訊號分成多個 field，每個 field 最多2個訊號以確保不超過 Discord 1024 字符限制
        signals_per_field = 2
        for i in range(0, len(top_signals), signals_per_field):
            field_signals = top_signals[i:i+signals_per_field]
            field_content = "\n\n".join(field_signals)
            
            # 設定 field 名稱
            start_num = i + 1
            end_num = min(i + signals_per_field, len(top_signals))
            
            if i == 0:
                field_name = f"🏆 TOP 做多推薦 ({start_num}-{end_num})"
            else:
                field_name = f"📈 做多推薦 ({start_num}-{end_num})"
            
            main_embed.add_field(
                name=field_name,
                value=field_content,
                inline=False
            )

    # 添加說明
    main_embed.add_field(
        name="ℹ️ 做多訊號評分說明",
        value="```\n🚀 技術突破: 25分 | ⬆️ 技術反彈: 15分\n🔥 機構看漲訊號: 依強度評分\n🏗️ 突破確認: 15分 | 趨勢轉變: 20分\n📦 機構大單區: 最高15分\n💎 價格缺口: 最高10分\n⚡ 大戶洗盤: 最高10分\n💰 借貸年利率: 最高10分```",
        inline=False
    )
    
    # 添加 SMC 價格區間說明
    main_embed.add_field(
        name="📊 SMC 價格區間說明",
        value="```\n🔴 高價區: 70%-100% 價格範圍 (賣出區域)\n🟢 低價區: 0%-30% 價格範圍 (買入區域)\n🟡 平衡區: 30%-70% 價格範圍 (觀望區域)\n\n基於過去100根K線的高低點計算\n適合設定止盈止損參考點位```",
        inline=False
    )
    
    main_embed.set_footer(text="⚠️ 僅供參考，請自行評估風險 | 專注做多機會分析")
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
    """獲取 SMC 分析亮點（中文化）"""
    if not smc_data:
        return "基礎分析", {}
    
    highlights = []
    zone_info = {}
    
    # 檢查市場結構
    structure = smc_data.get('market_structure', {})
    if structure.get('bos_signals'):
        highlights.append('突破確認')
    if structure.get('choch_signals'):
        highlights.append('趨勢轉變')
    
    # 檢查 Order Blocks (機構大單區)
    order_blocks = smc_data.get('order_blocks', [])
    if order_blocks:
        active_obs = [ob for ob in order_blocks if ob.get('active', False)]
        if active_obs:
            highlights.append(f'{len(active_obs)}個大單區')
    
    # 檢查 Fair Value Gaps (價格缺口)
    fvgs = smc_data.get('fair_value_gaps', [])
    if fvgs:
        highlights.append(f'{len(fvgs)}個價格缺口')
    
    # 檢查流動性掃蕩
    sweeps = smc_data.get('liquidity_sweeps', [])
    if sweeps:
        highlights.append('大戶洗盤')
    
    # 檢查當前區域並獲取價格信息
    zones = smc_data.get('premium_discount', {})
    current_zone = zones.get('current_zone', '')
    if current_zone:
        zone_map = {
            'PREMIUM': '🔴高價區', 
            'DISCOUNT': '🟢低價區', 
            'EQUILIBRIUM': '🟡平衡區'
        }
        zone_name = zone_map.get(current_zone, '')
        if zone_name:
            highlights.append(zone_name)
            
        # 提取價格區間信息
        premium_zone = zones.get('premium_zone', {})
        discount_zone = zones.get('discount_zone', {})
        if premium_zone and discount_zone:
            zone_info = {
                'high_price_zone': f"${premium_zone.get('start', 0):.6f}-${premium_zone.get('end', 0):.6f}",
                'low_price_zone': f"${discount_zone.get('start', 0):.6f}-${discount_zone.get('end', 0):.6f}",
                'current_zone': zone_name
            }
    
    return ' | '.join(highlights) if highlights else '基礎分析', zone_info

def format_score_breakdown(score_factors):
    """格式化評分明細（中文化）"""
    if not score_factors:
        return "無評分資料"
    
    breakdown_parts = []
    
    # Vegas 通道評分
    if 'vegas_breakout' in score_factors:
        breakdown_parts.append(f"技術突破: {score_factors['vegas_breakout']}分")
    elif 'vegas_bounce' in score_factors:
        breakdown_parts.append(f"技術反彈: {score_factors['vegas_bounce']}分")
    
    # SMC 結構評分
    if 'smc_bos' in score_factors:
        breakdown_parts.append(f"突破確認: {score_factors['smc_bos']}分")
    if 'smc_choch' in score_factors:
        breakdown_parts.append(f"趨勢轉變: {score_factors['smc_choch']}分")
    
    # Order Blocks 評分
    if 'order_blocks' in score_factors:
        breakdown_parts.append(f"大單區: {score_factors['order_blocks']}分")
    
    # Fair Value Gaps 評分
    if 'fair_value_gaps' in score_factors:
        breakdown_parts.append(f"價格缺口: {score_factors['fair_value_gaps']}分")
    
    # 流動性掃蕩評分
    if 'liquidity_sweeps' in score_factors:
        breakdown_parts.append(f"大戶洗盤: {score_factors['liquidity_sweeps']}分")
    
    # APR 評分
    if 'high_apr' in score_factors:
        breakdown_parts.append(f"高年利率: {score_factors['high_apr']}分")
    elif 'medium_apr' in score_factors:
        breakdown_parts.append(f"中年利率: {score_factors['medium_apr']}分")
    elif 'low_apr' in score_factors:
        breakdown_parts.append(f"低年利率: {score_factors['low_apr']}分")
    
    return " | ".join(breakdown_parts) if breakdown_parts else "基礎評分"

@bot.event
async def on_ready():
    print(f'已登入 Discord: {bot.user}')
    # Run the signal collection task with promising filter enabled
    await send_enhanced_signals(filter_promising=True)
    # Once the task is complete, close the bot gracefully
    await bot.close()

if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)