import os
import asyncio
import aiohttp
import pandas as pd
from tqdm import tqdm
import discord
from discord.ext import commands
from dotenv import load_dotenv
from aiolimiter import AsyncLimiter
from smc_indicators_optimized import OptimizedSmartMoneyConceptsAnalyzer as SmartMoneyConceptsAnalyzer

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

async def get_funding_rate_async(session, symbol: str):
    """獲取最新的資金費率"""
    async with semaphore:
        async with token_rate_limiter:
            base_url = "https://api.gateio.ws/api/v4"
            url = f"{base_url}/futures/usdt/funding_rate"
            params = {
                'contract': symbol,
                'limit': 1
            }
            try:
                async with session.get(url, params=params) as r:
                    r.raise_for_status()
                    data = await r.json()
                    if data:
                        rate_info = data[0]
                        funding_rate = float(rate_info['r'])
                        return funding_rate
                    return None
            except Exception as e:
                # print(f"取得 {symbol} 資金費率失敗: {e}")
                return None

async def get_oi_growth_rate_async(session, symbol: str):
    """獲取OI增長率"""
    async with semaphore:
        async with token_rate_limiter:
            base_url = "https://api.gateio.ws/api/v4"
            url = f"{base_url}/futures/usdt/contract_stats"
            params = {
                'contract': symbol,
                'interval': '1h',
                'limit': 2
            }
            try:
                async with session.get(url, params=params) as r:
                    r.raise_for_status()
                    data = await r.json()
                    if data and len(data) >= 2:
                        prev_stat = data[1]
                        curr_stat = data[0]
                        
                        prev_oi = prev_stat.get('open_interest', 0)
                        curr_oi = curr_stat.get('open_interest', 0)
                        
                        if prev_oi > 0:
                            oi_change = curr_oi - prev_oi
                            oi_growth_rate = (oi_change / prev_oi) * 100
                            return {
                                'growth_rate': oi_growth_rate,
                                'current_oi': curr_oi,
                                'current_oi_usd': curr_stat.get('open_interest_usd')
                            }
                    return None
            except Exception as e:
                # print(f"取得 {symbol} OI增長率失敗: {e}")
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

def calculate_price_skew_indicator(df, lookback_period=100):
    """
    計算價格偏斜指標 - 識別偶爾出現超長上影線的幣種
    
    Args:
        df: K線數據 DataFrame (需包含 high, low, close 欄位)
        lookback_period: 回溯期間 (預設100根K線)
    
    Returns:
        dict: 包含各種偏斜指標的字典
    """
    if len(df) < lookback_period:
        return None
    
    # 取最近的數據
    recent_df = df.tail(lookback_period).copy()
    
    # 1. 基本統計指標
    high_prices = recent_df['high']
    close_prices = recent_df['close']
    
    # 計算平均值和中位數
    high_mean = high_prices.mean()
    high_median = high_prices.median()
    close_mean = close_prices.mean()
    close_median = close_prices.median()
    
    # 2. 核心偏斜指標
    # High價格的平均值/中位數比率 (核心概念)
    high_skew_ratio = high_mean / high_median if high_median > 0 else 1.0
    close_skew_ratio = close_mean / close_median if close_median > 0 else 1.0
    
    # 3. 上影線分析
    recent_df['upper_shadow'] = recent_df['high'] - recent_df[['open', 'close']].max(axis=1)
    recent_df['body_size'] = abs(recent_df['close'] - recent_df['open'])
    recent_df['total_range'] = recent_df['high'] - recent_df['low']
    
    # 上影線比例 (上影線/總範圍)
    recent_df['upper_shadow_ratio'] = recent_df['upper_shadow'] / recent_df['total_range']
    recent_df['upper_shadow_ratio'] = recent_df['upper_shadow_ratio'].fillna(0)
    
    # 4. 識別極端上影線
    # 上影線長度超過K線總高度50%的情況
    extreme_upper_shadows = recent_df[recent_df['upper_shadow_ratio'] > 0.5]
    extreme_shadow_count = len(extreme_upper_shadows)
    extreme_shadow_frequency = extreme_shadow_count / lookback_period
    
    # 5. 計算上影線的統計特徵
    upper_shadow_mean = recent_df['upper_shadow_ratio'].mean()
    upper_shadow_median = recent_df['upper_shadow_ratio'].median()
    upper_shadow_std = recent_df['upper_shadow_ratio'].std()
    
    # 6. 價格跳躍檢測
    recent_df['price_change'] = recent_df['high'].pct_change()
    extreme_jumps = recent_df[recent_df['price_change'] > 0.1]  # 10%以上的跳躍
    jump_frequency = len(extreme_jumps) / lookback_period
    
    # 7. 整體偏斜評分 (0-100)
    skew_score = 0
    
    # High價格偏斜評分 (最高40分)
    if high_skew_ratio > 1.2:  # 平均值比中位數高20%以上
        skew_score += min(40, (high_skew_ratio - 1) * 100)
    
    # 極端上影線頻率評分 (最高30分)
    if extreme_shadow_frequency > 0.05:  # 超過5%的K線有極端上影線
        skew_score += min(30, extreme_shadow_frequency * 600)
    
    # 價格跳躍頻率評分 (最高20分)
    if jump_frequency > 0.02:  # 超過2%的K線有大幅跳躍
        skew_score += min(20, jump_frequency * 1000)
    
    return {
        'high_skew_ratio': high_skew_ratio,
        'close_skew_ratio': close_skew_ratio,
        'high_mean': high_mean,
        'high_median': high_median,
        'extreme_shadow_count': extreme_shadow_count,
        'extreme_shadow_frequency': extreme_shadow_frequency,
        'upper_shadow_mean': upper_shadow_mean,
        'upper_shadow_median': upper_shadow_median,
        'jump_frequency': jump_frequency,
        'skew_score': min(100, skew_score),
        'is_skew_candidate': skew_score >= 30,  # 30分以上認為是偏斜候選
        'skew_level': get_skew_level(skew_score)
    }

def get_skew_level(score):
    """根據評分判定偏斜等級"""
    if score >= 60:
        return "極度偏斜"
    elif score >= 40:
        return "高度偏斜"
    elif score >= 20:
        return "中度偏斜"
    else:
        return "正常分佈"

def calculate_signal_score(vegas_signal, smc_analysis, symbol, apr_data, price_skew_data=None, funding_rate=None, oi_data=None):
    """計算綜合訊號評分 (0-100) - 增強版包含爆發性模式檢測和價格偏斜分析"""
    score = 0
    factors = {}
    explosive_indicators = []  # 記錄爆發性指標
    
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
            score += 15  # 突破訊號較強
            factors['vegas_breakout'] = 15
            explosive_indicators.append('強勢突破')
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
            explosive_indicators.append('結構突破')
        if structure['choch_signals']:
            score += 20  # 趨勢轉變更重要
            factors['smc_choch'] = 20
            explosive_indicators.append('趨勢轉變')
    
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
            explosive_indicators.append('超高APY')
        elif apr_data > 0.5:  # 50%+
            score += 6
            factors['medium_apr'] = 6
            explosive_indicators.append('高APY')
        elif apr_data > 0.2:  # 20%+
            score += 3
            factors['low_apr'] = 3
    
    # 💡 價格偏斜指標加成 (0-15分) - 新增功能
    if price_skew_data:
        skew_score = price_skew_data.get('skew_score', 0)
        if skew_score >= 60:  # 極度偏斜
            score += 15
            factors['extreme_skew'] = 15
            explosive_indicators.append('極端偏斜')
        elif skew_score >= 40:  # 高度偏斜
            score += 10
            factors['high_skew'] = 10
            explosive_indicators.append('高度偏斜')
        elif skew_score >= 20:  # 中度偏斜
            score += 5
            factors['medium_skew'] = 5
            explosive_indicators.append('中度偏斜')
        
        # 記錄偏斜數據供顯示用
        factors['skew_data'] = price_skew_data
    
    # 🚀 爆發性模式檢測和加成
    is_explosive = detect_explosive_pattern(explosive_indicators, smc_analysis, score, funding_rate, oi_data)
    if is_explosive:
        score += 20  # 爆發性模式額外加分
        factors['explosive_bonus'] = 20
        factors['explosive_indicators'] = explosive_indicators
        factors['is_explosive'] = True
    else:
        factors['is_explosive'] = False
    
    return min(100, score), factors

def detect_explosive_pattern(indicators, smc_analysis, base_score, funding_rate=None, oi_data=None):
    """檢測是否為爆發性模式 - 優化版：重視資金費率、OI增長、價格偏斜、APY以及低價區"""
    explosive_score = 0
    
    # 1. 資金費率指標 (新增重點: 30分)
    if funding_rate is not None:
        if funding_rate > 0.01:  # >1% 極高資金費率 - 強烈多頭情緒
            explosive_score += 30
            indicators.append('極高資金費率')
        elif funding_rate > 0.005:  # >0.5% 高資金費率
            explosive_score += 20  
            indicators.append('高資金費率')
        elif funding_rate > 0.002:  # >0.2% 中等資金費率
            explosive_score += 10
            indicators.append('中等資金費率')
        elif funding_rate < -0.005:  # <-0.5% 極低資金費率 - 可能逆轉機會
            explosive_score += 15
            indicators.append('極低資金費率')
    
    # 2. OI增長率指標 (新增重點: 25分)
    if oi_data and isinstance(oi_data, dict):
        growth_rate = oi_data.get('growth_rate', 0)
        if growth_rate > 20:  # >20% 極高OI增長
            explosive_score += 25
            indicators.append('極高OI增長')
        elif growth_rate > 10:  # >10% 高OI增長
            explosive_score += 20
            indicators.append('高OI增長')
        elif growth_rate > 5:  # >5% 中等OI增長
            explosive_score += 15
            indicators.append('中等OI增長')
        elif growth_rate > 1:  # >1% 輕微OI增長
            explosive_score += 10
            indicators.append('輕微OI增長')
    
    # 3. 低價區判斷 (新增重點: 25分)
    if smc_analysis:
        zones = smc_analysis.get('premium_discount', {})
        current_zone = zones.get('current_zone', '')
        if current_zone == 'DISCOUNT':  # 在低價區 - 更容易爆發
            explosive_score += 25
            indicators.append('低價區位置')
        elif current_zone == 'EQUILIBRIUM':  # 在平衡區
            explosive_score += 10
            indicators.append('平衡區位置')
        # 高價區不加分，反而是風險信號
    
    # 4. APY指標 (保持重要: 25分)
    if '超高APY' in indicators:
        explosive_score += 25
    elif '高APY' in indicators:
        explosive_score += 20
    
    # 5. 價格偏斜指標 (保持重要: 25分)
    if '極端偏斜' in indicators:
        explosive_score += 25
    elif '高度偏斜' in indicators:
        explosive_score += 20
    elif '中度偏斜' in indicators:
        explosive_score += 10
    
    # 6. 技術指標 (降低權重: 15分)
    if '強勢突破' in indicators:
        explosive_score += 5
    if '結構突破' in indicators:
        explosive_score += 5
    if '趨勢轉變' in indicators:
        explosive_score += 5
    
    # 7. SMC多重確認 (保持: 15分)
    if smc_analysis:
        confirmation_count = 0
        if smc_analysis.get('order_blocks'):
            confirmation_count += 1
        if smc_analysis.get('fair_value_gaps'):
            confirmation_count += 1
        if smc_analysis.get('liquidity_sweeps'):
            confirmation_count += 1
        
        explosive_score += confirmation_count * 5  # 每個確認5分
    
    # 8. 複合指標加成 (新增)
    # 資金費率 + OI增長 組合 (最強信號)
    has_high_funding = any(x in indicators for x in ['極高資金費率', '高資金費率'])
    has_oi_growth = any(x in indicators for x in ['極高OI增長', '高OI增長'])
    has_low_zone = '低價區位置' in indicators
    has_good_apy = any(x in indicators for x in ['超高APY', '高APY'])
    has_good_skew = any(x in indicators for x in ['極端偏斜', '高度偏斜'])
    
    # 三重組合加成
    if has_high_funding and has_oi_growth and has_low_zone:
        explosive_score += 20  # 資金費率 + OI增長 + 低價區 = 完美組合
        indicators.append('完美三重組合')
    elif (has_high_funding and has_oi_growth) or (has_good_apy and has_good_skew):
        explosive_score += 15  # 雙重組合
        indicators.append('雙重組合')
    
    # 爆發性模式判定：需要達到60分以上
    return explosive_score >= 60

def enhance_vegas_with_smc(df, symbol):
    """使用 SMC 增強 Vegas 通道分析 - 加入價格偏斜分析"""
    if df is None or len(df) < 676:
        return None
    
    # 🔍 計算價格偏斜指標
    price_skew_data = calculate_price_skew_indicator(df)
    
    # 獲取 Vegas 訊號
    vegas_df = detect_vegas_turning_points(df)
    
    # 獲取 SMC 分析
    try:
        smc_analysis = smc_analyzer.get_comprehensive_analysis(df)
    except Exception as e:
        print(f"SMC 分析失敗 {symbol}: {e}")
        smc_analysis = {}
    
    if vegas_df is None or vegas_df.empty:
        # 即使沒有 Vegas 訊號，也檢查是否有強 SMC 訊號或偏斜特徵
        has_strong_smc = False
        has_strong_skew = False
        
        if smc_analysis:
            structure = smc_analysis.get('market_structure', {})
            if (structure.get('bos_signals') or structure.get('choch_signals') or
                smc_analysis.get('order_blocks') or smc_analysis.get('liquidity_sweeps')):
                has_strong_smc = True
        
        if price_skew_data and price_skew_data.get('is_skew_candidate', False):
            has_strong_skew = True
        
        if has_strong_smc or has_strong_skew:
            # 創建純 SMC/偏斜 訊號
            current_price = df['close'].iloc[-1]
            signal_type = 'SMC_BULLISH' if smc_analysis.get('overall_bias') == 'BULLISH' else 'SMC_BEARISH'
            
            # 如果有強偏斜特徵，優先顯示偏斜信號
            if has_strong_skew:
                signal_type = f"SKEW_{signal_type}"
            
            result_df = pd.DataFrame({
                'close': [current_price],
                'vegas_signal': [signal_type],
                'ema12': [df['close'].ewm(span=12).mean().iloc[-1]],
                'vegas_high': [0],
                'vegas_low': [0]
            })
            result_df['smc_analysis'] = [smc_analysis]
            result_df['price_skew_data'] = [price_skew_data]
            result_df['signal_source'] = ['SMC_SKEW'] if has_strong_skew else ['SMC_ONLY']
            return result_df
        return None
    
    # 如果有 Vegas 訊號，加入 SMC 分析和偏斜數據
    vegas_df = vegas_df.copy()
    vegas_df['smc_analysis'] = [smc_analysis]
    vegas_df['price_skew_data'] = [price_skew_data]
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

        # 先進行基礎評分，篩選出有潛力的訊號
        print("🔍 正在計算基礎技術評分...")
        temp_scores = []
        temp_factors = []
        
        for idx, row in final_df.iterrows():
            smc_analysis = row.get('smc_analysis', {})
            apr_data = row.get('compound_apr', 0)
            price_skew_data = row.get('price_skew_data', {})
            
            funding_rate = row.get('funding_rate')
            oi_data = row.get('oi_data')
            score, factors = calculate_signal_score(row, smc_analysis, row['symbol'], apr_data, price_skew_data, funding_rate, oi_data)
            temp_scores.append(score)
            temp_factors.append(factors)
        
        final_df['temp_score'] = temp_scores
        final_df['temp_factors'] = temp_factors
        
        # 只對高分訊號（例如>40分）獲取資金費率和OI數據，減少API調用
        high_score_df = final_df[final_df['temp_score'] > 30].copy()  # 降低門檻以確保有足夠數據
        
        if len(high_score_df) > 0:
            print(f"📊 正在為{len(high_score_df)}個高分訊號獲取資金費率和OI增長率數據...")
            high_score_symbols = high_score_df['symbol'].unique()
            
            # 獲取資金費率
            funding_tasks = [get_funding_rate_async(session, symbol) for symbol in high_score_symbols]
            funding_results = await asyncio.gather(*funding_tasks, return_exceptions=True)
            funding_map = {}
            for symbol, result in zip(high_score_symbols, funding_results):
                if not isinstance(result, Exception) and result is not None:
                    funding_map[symbol] = result
            
            # 獲取OI增長率
            oi_tasks = [get_oi_growth_rate_async(session, symbol) for symbol in high_score_symbols]
            oi_results = await asyncio.gather(*oi_tasks, return_exceptions=True)
            oi_map = {}
            for symbol, result in zip(high_score_symbols, oi_results):
                if not isinstance(result, Exception) and result is not None:
                    oi_map[symbol] = result
            
            # 添加到DataFrame（所有數據，但只有高分的有值）
            final_df['funding_rate'] = final_df['symbol'].apply(lambda x: funding_map.get(x))
            final_df['oi_data'] = final_df['symbol'].apply(lambda x: oi_map.get(x))
        else:
            print("⚠️ 沒有找到高分訊號，跳過資金費率和OI數據獲取")
            final_df['funding_rate'] = None
            final_df['oi_data'] = None
        
        # 使用已計算的評分，清理臨時欄位
        final_df['signal_score'] = final_df['temp_score']
        final_df['score_factors'] = final_df['temp_factors']
        final_df['smc_data'] = final_df['smc_analysis']
        
        # 清理臨時欄位
        final_df = final_df.drop(['temp_score', 'temp_factors'], axis=1)

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

    # 只篩選做多訊號 (包含新的偏斜信號)
    long_signals = ['LONG_BREAKOUT', 'LONG_BOUNCE', 'SMC_BULLISH', 'SKEW_SMC_BULLISH']
    final_df = final_df[final_df['vegas_signal'].isin(long_signals)]
    
    # 創建綜合排序分數，考慮資金費率和OI增長率
    def calculate_sort_score(row):
        base_score = row['signal_score']
        
        # 資金費率加分項（正值表示多頭情緒）
        funding_bonus = 0
        if pd.notna(row['funding_rate']):
            funding_rate = row['funding_rate']
            if funding_rate > 0:  # 正資金費率（多頭支付空頭）
                if funding_rate > 0.01:  # >1%極高
                    funding_bonus = 15
                elif funding_rate > 0.005:  # >0.5%高
                    funding_bonus = 10
                elif funding_rate > 0.001:  # >0.1%中等
                    funding_bonus = 5
            # 負資金費率可能表示逆向機會，給小加分
            elif funding_rate < -0.001:  # <-0.1%
                funding_bonus = 3
        
        # OI增長率加分項（正值表示資金流入）
        oi_bonus = 0
        if pd.notna(row['oi_data']) and isinstance(row['oi_data'], dict):
            growth_rate = row['oi_data'].get('growth_rate', 0)
            if growth_rate > 10:  # >10%極高增長
                oi_bonus = 15
            elif growth_rate > 5:  # >5%高增長
                oi_bonus = 10
            elif growth_rate > 1:  # >1%中等增長
                oi_bonus = 5
            elif growth_rate > 0:  # 正增長
                oi_bonus = 2
        
        return base_score + funding_bonus + oi_bonus
    
    # 計算綜合排序分數
    final_df['sort_score'] = final_df.apply(calculate_sort_score, axis=1)
    
    # 按綜合排序分數排序，取TOP 10
    final_df = final_df.sort_values(by='sort_score', ascending=False).head(10)

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
            score_factors = row.get('score_factors', {})
            is_explosive = score_factors.get('is_explosive', False)
            
            signal_emoji = get_signal_emoji(signal_type, is_explosive)
            signal_name = get_signal_name(signal_type)
            apr_str = f"{row['compound_apr']:.2%}" if pd.notna(row['compound_apr']) else "N/A"
            score = row['signal_score']
            
            # 獲取 SMC 亮點和價格區間
            smc_highlights, zone_info = get_smc_highlights(row.get('smc_data', {}))
            
            # 獲取評分明細
            score_breakdown = format_score_breakdown(score_factors)
            
            # 根據排名添加獎牌emoji
            rank_emoji = ""
            if i == 1:
                rank_emoji = "🥇 "
            elif i == 2:
                rank_emoji = "🥈 "
            elif i == 3:
                rank_emoji = "🥉 "
            
            # 爆發性模式特殊標註
            explosive_tag = ""
            if is_explosive:
                explosive_indicators = score_factors.get('explosive_indicators', [])
                explosive_tag = f"\n     🚨 **爆發性模式**: {' | '.join(explosive_indicators)} 🚨"
            
            # 格式化價格區間信息（簡化格式以節省字符）
            zone_display = ""
            if zone_info:
                zone_display = f"\n     🔴`{zone_info['high_price_zone']}` 🟢`{zone_info['low_price_zone']}`"
            
            # 獲取資金費率和OI增長率信息
            funding_rate = row.get('funding_rate')
            oi_data = row.get('oi_data')
            
            funding_str = f"{funding_rate:.4%}" if funding_rate is not None else "N/A"
            oi_growth_str = "N/A"
            if oi_data and isinstance(oi_data, dict):
                growth_rate = oi_data.get('growth_rate', 0)
                oi_growth_str = f"{growth_rate:+.2f}%"
            
            # 獲取綜合排序分數
            sort_score = row.get('sort_score', score)
            sort_bonus = sort_score - score
            
            # 資金費率和OI增長率的特殊標註
            funding_tag = ""
            oi_tag = ""
            
            if funding_rate is not None:
                if funding_rate > 0.01:
                    funding_tag = " 🔥"
                elif funding_rate > 0.005:
                    funding_tag = " ⬆️"
                elif funding_rate < -0.001:
                    funding_tag = " 🔄"
            
            if oi_data and isinstance(oi_data, dict):
                growth_rate = oi_data.get('growth_rate', 0)
                if growth_rate > 10:
                    oi_tag = " 🚀"
                elif growth_rate > 5:
                    oi_tag = " 📈"
                elif growth_rate > 1:
                    oi_tag = " ⬆️"
            
            top_signals.append(
                f"{rank_emoji}`{i}.` **{row['symbol']}** {signal_emoji} `{score:.0f}分` `(+{sort_bonus:.0f})`\n"
                f"     💰 `${row['close']:.6f}` | 📊 `{signal_name}` | 🏦 `{apr_str}`\n"
                f"     💸 資金費率: `{funding_str}`{funding_tag} | 📈 OI增長: `{oi_growth_str}`{oi_tag}\n"
                f"     🎯 {smc_highlights}{zone_display}{explosive_tag}\n"
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
        value="```\n🚀 技術突破: 25分 | ⬆️ 技術反彈: 15分\n🔥 機構看漲訊號: 依強度評分\n🏗️ 突破確認: 15分 | 趨勢轉變: 20分\n📦 機構大單區: 最高15分\n💎 價格缺口: 最高10分\n⚡ 大戶洗盤: 最高10分\n💰 借貸年利率: 最高10分\n📊 價格偏斜指標: 最高15分 (NEW!)\n💥 爆發性模式: 額外20分```",
        inline=False
    )
    
    # 爆發性模式特別說明 (2024優化版)
    main_embed.add_field(
        name="💥 爆發性模式說明 (重點關注!)",
        value="```\n🔥 新識別標準 (資金面為王):\n• 💰 資金費率 >0.5% (多頭情緒強烈)\n• 📈 OI增長率 >5% (大資金湧入)\n• 🎯 低價區位置 (SMC DISCOUNT區域)\n• 💎 高APY + 價格偏斜 (基本面確認)\n\n🚀 完美組合 (必爆信號):\n• 資金費率 + OI增長 + 低價區 = 三重確認\n• 評分門檻: ≥60分 = 爆發性模式\n\n💎 特徵 (不變):\n• 短期內可能暴漲50%-300%\n• 快進快出，建議1-2週內止盈\n• 小幣爆發的核心機會\n\n⚠️ 風險控制:\n• 波動性極大，嚴格風控\n• 優先關注低價區+高資金費率組合```",
        inline=False
    )
    
    # 新增價格偏斜指標說明
    main_embed.add_field(
        name="📊 價格偏斜指標說明 (NEW!)",
        value="```\n🎯 核心概念:\n• 識別平時穩定但偶爾出現超長上影線的幣種\n• 平均值/中位數比值 > 1.2 = 有偏斜特徵\n• 極端上影線頻率 > 5% = 高偏斜\n\n💡 交易意義:\n• 這類幣種往往在機構試探後大漲\n• 適合潛伏等待突破性機會\n• 風險相對較低但爆發力強\n\n📈 評分標準:\n• 極度偏斜(60+分): 15分 | 高度偏斜(40+分): 10分\n• 中度偏斜(20+分): 5分```",
        inline=False
    )
    
    # 添加 SMC 價格區間說明
    main_embed.add_field(
        name="📊 SMC 價格區間說明",
        value="```\n🔴 高價區: 70%-100% 價格範圍 (賣出區域)\n🟢 低價區: 0%-30% 價格範圍 (買入區域)\n🟡 平衡區: 30%-70% 價格範圍 (觀望區域)\n\n基於過去100根K線的高低點計算\n適合設定止盈止損參考點位```",
        inline=False
    )
    
    # 添加資金費率和OI增長率說明
    main_embed.add_field(
        name="💸 資金費率 & 📈 OI增長率說明 (NEW!)",
        value="```\n💸 資金費率 (Funding Rate):\n• 永續合約多空平衡指標\n• 正值: 多頭支付空頭 (看漲情緒)\n• 負值: 空頭支付多頭 (看跌情緒)\n• 🔥>1% 極高 (+15分) | ⬆️>0.5% 高 (+10分)\n• 🔄<-0.1% 逆向機會 (+3分)\n\n📈 OI增長率 (Open Interest Growth):\n• 未平倉合約量變化\n• 正值: 新資金流入，趨勢可能延續\n• 🚀>10% 極高 (+15分) | 📈>5% 高 (+10分)\n• ⬆️>1% 中等 (+5分) | >0% 小幅 (+2分)```",
        inline=False
    )
    
    # 添加綜合排序說明
    main_embed.add_field(
        name="🎯 綜合排序說明 (NEW!)",
        value="```\n排序邏輯:\n基礎技術評分 + 資金費率加分 + OI增長率加分\n\n顯示格式:\n• 基礎分數: 技術分析綜合評分\n• (+加分): 資金費率和OI增長率額外加分\n• 特殊標誌: 🔥🚀📈⬆️🔄 表示不同等級\n\n優勢:\n• 優先推薦有資金流入和多頭情緒的幣種\n• 結合技術面和資金面的雙重確認\n• 提高爆發性機會的識別準確度```",
        inline=False
    )
    
    main_embed.set_footer(text="⚠️ 僅供參考，請自行評估風險 | 專注做多機會分析")
    main_embed.timestamp = discord.utils.utcnow()

    await channel.send(embed=main_embed)

def get_signal_emoji(signal_type, is_explosive=False):
    """獲取訊號對應的 emoji - 爆發性模式特殊標註"""
    if is_explosive:
        # 爆發性模式專用emoji
        explosive_emoji_map = {
            'LONG_BREAKOUT': '💥🚀',
            'LONG_BOUNCE': '💥⬆️', 
            'SMC_BULLISH': '💥🔥',
            'SKEW_SMC_BULLISH': '💥📊🔥',  # 新增偏斜信號
            'SHORT_BREAKDOWN': '💥📉',
            'SHORT_FAILED_BOUNCE': '💥⬇️',
            'SMC_BEARISH': '💥❄️'
        }
        return explosive_emoji_map.get(signal_type, '💥❓')
    else:
        # 一般訊號emoji
        emoji_map = {
            'LONG_BREAKOUT': '🚀',
            'LONG_BOUNCE': '⬆️',
            'SHORT_BREAKDOWN': '📉',
            'SHORT_FAILED_BOUNCE': '⬇️',
            'SMC_BULLISH': '🔥',
            'SKEW_SMC_BULLISH': '📊🔥',  # 新增偏斜信號
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
        'SKEW_SMC_BULLISH': '偏斜看漲',  # 新增偏斜信號
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
    else:
        highlights.append('無價格缺口')
    
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
    """格式化評分明細（中文化）- 包含爆發性模式和價格偏斜指標"""
    if not score_factors:
        return "無評分資料"
    
    breakdown_parts = []
    
    # 爆發性模式優先顯示
    if score_factors.get('is_explosive', False):
        breakdown_parts.append(f"💥爆發性: {score_factors.get('explosive_bonus', 20)}分")
    
    # 價格偏斜指標 (新增功能，優先顯示)
    if 'extreme_skew' in score_factors:
        breakdown_parts.append(f"📊極端偏斜: {score_factors['extreme_skew']}分")
    elif 'high_skew' in score_factors:
        breakdown_parts.append(f"📊高度偏斜: {score_factors['high_skew']}分")
    elif 'medium_skew' in score_factors:
        breakdown_parts.append(f"📊中度偏斜: {score_factors['medium_skew']}分")
    
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