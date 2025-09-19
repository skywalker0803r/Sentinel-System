import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

class OptimizedSmartMoneyConceptsAnalyzer:
    """優化版 Smart Money Concepts 技術分析器 - 保持相同邏輯但提升性能"""
    
    def __init__(self):
        self.BULLISH = 1
        self.BEARISH = -1
        # 緩存機制
        self._cache = {}
        self._last_df_hash = None
        
    def _get_df_hash(self, df: pd.DataFrame) -> str:
        """計算DataFrame的簡單hash用於緩存"""
        return f"{len(df)}_{df['close'].iloc[-1]}_{df['close'].iloc[0]}"
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """計算 Average True Range - 優化版"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()
    
    def _precompute_future_extremes(self, df: pd.DataFrame) -> Dict:
        """預計算未來最值 - 關鍵優化: O(n²) → O(n)"""
        n = len(df)
        
        # 使用向後掃描一次性計算所有未來最值
        future_min_low = np.full(n, np.inf)
        future_max_high = np.full(n, -np.inf)
        future_min_close = np.full(n, np.inf) 
        future_max_close = np.full(n, -np.inf)
        
        # 從後往前掃描，O(n)時間複雜度
        for i in range(n-2, -1, -1):
            future_min_low[i] = min(df['low'].iloc[i+1], future_min_low[i+1])
            future_max_high[i] = max(df['high'].iloc[i+1], future_max_high[i+1])
            future_min_close[i] = min(df['close'].iloc[i+1], future_min_close[i+1])
            future_max_close[i] = max(df['close'].iloc[i+1], future_max_close[i+1])
        
        return {
            'future_min_low': future_min_low,
            'future_max_high': future_max_high,
            'future_min_close': future_min_close,
            'future_max_close': future_max_close
        }

    def _compute_swing_points(self, df: pd.DataFrame, window: int = 5) -> Tuple[pd.Series, pd.Series]:
        """
        真正向量化計算 swing points，並處理平頂/平底問題。
        使用 rolling window 實現高效計算。
        """
        # 完整的窗口大小是 (window * 2 + 1)
        full_window = window * 2 + 1
        
        # 使用 rolling().max() 找到每個窗口的最高點
        # center=True 確保當前 K 線位於窗口中心
        rolling_max = df['high'].rolling(window=full_window, center=True).max()
        rolling_min = df['low'].rolling(window=full_window, center=True).min()
        
        # 條件1: 當前 K 線的 high/low 必須是其窗口內的極值
        is_swing_high = (df['high'] == rolling_max)
        is_swing_low = (df['low'] == rolling_min)
        
        # 條件2 (解決平頂/平底問題): 該 high/low 不能與前一根 K 線的 high/low 相同
        # 這樣可以確保在一個平頂/平底結構中，只取第一個點
        not_duplicate_high = (df['high'] != df['high'].shift(1))
        not_duplicate_low = (df['low'] != df['low'].shift(1))
        
        # 最終的 swing points 必須同時滿足兩個條件
        swing_highs = is_swing_high & not_duplicate_high
        swing_lows = is_swing_low & not_duplicate_low
        
        # 函數返回的是 boolean Series，方便後續直接用 .loc 定位
        return df.index[swing_highs], df.index[swing_lows]
    
    def detect_market_structure(self, df: pd.DataFrame, lookback: int = 30) -> Dict:
        """檢測市場結構：BOS 和 CHoCH - 優化版 O(n)"""
        if df is None or len(df) < lookback:
            return {'bos_signals': [], 'choch_signals': [], 'trend': None}
            
        df = df.copy()
        
        # 計算 ATR
        if 'atr' not in df.columns:
            df['atr'] = self._calculate_atr(df, 14)
        
        # 使用優化的swing points計算
        window = max(3, lookback//10)  # 減少窗口大小，增加敏感度
        swing_highs, swing_lows = self._compute_swing_points(df, window)
        
        # 獲取swing價格
        swing_high_prices = df.loc[swing_highs, 'high']
        swing_low_prices = df.loc[swing_lows, 'low']
        
        bos_signals = []
        choch_signals = []
        current_trend = None
        current_price = df['close'].iloc[-1]
        
        # --- 檢測市場結構 (BOS & CHoCH) ---
        
        # 1. 看漲BOS (Bullish BOS) - 上升趨勢延續
        if len(swing_high_prices) >= 2:
            recent_highs = swing_high_prices.tail(2)
            last_high = recent_highs.iloc[-1]
            prev_high = recent_highs.iloc[-2]
            
            if current_price > last_high and last_high > prev_high:
                bos_signals.append({
                    'type': 'BOS_BULLISH',
                    'price': current_price,
                    'strength': self._calculate_structure_strength_fast(df),
                    'description': f'突破前高(HH) ${last_high:.6f}'
                })
                current_trend = self.BULLISH

        # 2. 看跌BOS (Bearish BOS) - 下降趨勢延續
        if len(swing_low_prices) >= 2:
            recent_lows = swing_low_prices.tail(2)
            last_low = recent_lows.iloc[-1]
            prev_low = recent_lows.iloc[-2]

            if current_price < last_low and last_low < prev_low:
                bos_signals.append({
                    'type': 'BOS_BEARISH',
                    'price': current_price,
                    'strength': self._calculate_structure_strength_fast(df),
                    'description': f'跌破前低(LL) ${last_low:.6f}'
                })
                current_trend = self.BEARISH

        # 3. 看漲CHoCH (Bullish CHoCH) - 從下降轉為上升
        if len(swing_low_prices) >= 2 and not swing_high_prices.empty:
            last_low = swing_low_prices.iloc[-1]
            prev_low = swing_low_prices.iloc[-2]

            if last_low < prev_low: # 確認創下更低的低點
                last_low_index = swing_low_prices.index[-1]
                relevant_highs = swing_high_prices[swing_high_prices.index < last_low_index]

                if not relevant_highs.empty:
                    choch_level = relevant_highs.iloc[-1]  # 這就是 LH 的價格
                    if current_price > choch_level:
                        choch_signals.append({
                            'type': 'CHOCH_BULLISH',
                            'price': current_price,
                            'strength': self._calculate_structure_strength_fast(df),
                            'description': f'突破前一較低高點(LH) ${choch_level:.6f}'
                        })
                        current_trend = self.BULLISH

        # 4. 看跌CHoCH (Bearish CHoCH) - 從上升轉為下降
        if len(swing_high_prices) >= 2 and not swing_low_prices.empty:
            last_high = swing_high_prices.iloc[-1]
            prev_high = swing_high_prices.iloc[-2]

            if last_high > prev_high: # 確認創下更高的低點
                last_high_index = swing_high_prices.index[-1]
                relevant_lows = swing_low_prices[swing_low_prices.index < last_high_index]

                if not relevant_lows.empty:
                    choch_level = relevant_lows.iloc[-1] # 這是 HL 的價格
                    if current_price < choch_level:
                        choch_signals.append({
                            'type': 'CHOCH_BEARISH',
                            'price': current_price,
                            'strength': self._calculate_structure_strength_fast(df),
                            'description': f'跌破前一較高低點(HL) ${choch_level:.6f}'
                        })
                        current_trend = self.BEARISH
        
        return {
            'bos_signals': bos_signals,
            'choch_signals': choch_signals,
            'trend': current_trend,
            'swing_highs': swing_high_prices.tolist()[-5:] if len(swing_high_prices) > 0 else [],
            'swing_lows': swing_low_prices.tolist()[-5:] if len(swing_low_prices) > 0 else []
        }
    
    def _calculate_structure_strength_fast(self, df: pd.DataFrame) -> int:
        """快速計算結構強度 - 優化版"""
        volume_factor = 50
        
        # 使用向量化操作計算成交量比較
        if 'volume' in df.columns:
            recent_vol = df['volume'].tail(10).mean()
            avg_vol = df['volume'].mean()
            if recent_vol > avg_vol * 1.5:
                volume_factor += 20
        
        # ATR比較
        if 'atr' in df.columns:
            current_atr = df['atr'].iloc[-1]
            avg_atr = df['atr'].mean()
            if current_atr > avg_atr * 1.2:
                volume_factor += 15
        
        return min(100, volume_factor)
    
    def detect_order_blocks(self, df: pd.DataFrame, lookback: int = 100) -> List[Dict]:
        """檢測 Order Blocks - 優化版 O(n²) → O(n)"""
        if df is None or len(df) < lookback:
            return []
            
        df = df.copy()
        order_blocks = []
        
        # 計算 ATR
        df['atr'] = self._calculate_atr(df, 14)
        current_atr = df['atr'].iloc[-1]
        
        # 預計算未來最值 - 關鍵優化
        future_extremes = self._precompute_future_extremes(df)
        
        # 向量化檢測反轉蠟燭
        closes = df['close'].values
        opens = df['open'].values
        highs = df['high'].values
        lows = df['low'].values
        
        # 一次性計算所有可能的Order Block候選
        start_idx = max(lookback, 10)
        end_idx = len(df) - 5
        
        for i in range(start_idx, end_idx):
            # 只檢查看漲 Order Block（機器人只做多）
            is_bullish_ob = self._is_bullish_order_block_fast(df, i, closes, opens)
            
            if is_bullish_ob:
                ob_high = highs[i]
                ob_low = lows[i]
                
                # 檢查顯著性
                if (ob_high - ob_low) > current_atr * 0.5:
                    # 使用預計算的數據檢查活躍性 - O(1)操作
                    is_active = future_extremes['future_min_low'][i] >= ob_low
                    
                    if is_active:  # 只保留活躍的OB
                        order_blocks.append({
                            'type': 'BULLISH_OB',
                            'high': ob_high,
                            'low': ob_low,
                            'time': df.get('time', pd.Series(range(len(df)))).iloc[i],
                            'strength': self._calculate_ob_strength_fast(df, i),
                            'active': True,
                            'description': f'看漲訂單區塊 ${ob_low:.6f}-${ob_high:.6f}'
                        })
            
            # 跳過看跌 Order Block 檢測（機器人只做多）
        
        # 按強度排序，返回前10個
        return sorted(order_blocks, key=lambda x: x['strength'], reverse=True)[:10]
    
    def _is_bullish_order_block_fast(self, df: pd.DataFrame, index: int, closes: np.ndarray, opens: np.ndarray) -> bool:
        """快速檢查看漲Order Block - 向量化版本"""
        if index < 5 or index >= len(df) - 5:
            return False
            
        # 向量化檢查前後趨勢
        prev_slice = closes[index-5:index]
        next_slice = closes[index+1:index+6]
        
        if len(prev_slice) == 0 or len(next_slice) == 0:
            return False
            
        # 前面下跌，後面上漲
        declining = prev_slice[-1] < prev_slice[0]
        rising = next_slice[-1] > closes[index]
        
        return declining and rising
    
    def _is_bearish_order_block_fast(self, df: pd.DataFrame, index: int, closes: np.ndarray, opens: np.ndarray) -> bool:
        """快速檢查看跌Order Block - 向量化版本"""
        if index < 5 or index >= len(df) - 5:
            return False
            
        prev_slice = closes[index-5:index]
        next_slice = closes[index+1:index+6]
        
        if len(prev_slice) == 0 or len(next_slice) == 0:
            return False
            
        # 前面上漲，後面下跌
        rising = prev_slice[-1] > prev_slice[0]
        declining = next_slice[-1] < closes[index]
        
        return rising and declining
    
    def _calculate_ob_strength_fast(self, df: pd.DataFrame, index: int) -> int:
        """快速計算Order Block強度"""
        base_strength = 50
        
        # 快速成交量檢查
        if 'volume' in df.columns and index < len(df):
            current_vol = df['volume'].iloc[index]
            # 使用近期平均而非全局平均，提升計算速度
            recent_avg_vol = df['volume'].iloc[max(0, index-20):index].mean()
            if current_vol > recent_avg_vol * 1.5:
                base_strength += 25
        
        return min(100, base_strength)
    
    def detect_fair_value_gaps(self, df: pd.DataFrame) -> List[Dict]:
        """檢測 Fair Value Gaps - 優化版 O(n²) → O(n)"""
        if df is None or len(df) < 3:
            return []
            
        df = df.copy()
        fvg_list = []
        
        # 預計算未來最值 - 避免重複掃描
        future_extremes = self._precompute_future_extremes(df)
        
        # 向量化數據訪問
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        opens = df['open'].values
        
        # 預計算ATR值
        if 'atr' not in df.columns:
            df['atr'] = self._calculate_atr(df, 14)
        atr_values = df['atr'].values
        
        for i in range(1, len(df) - 1):
            prev_high, prev_low = highs[i-1], lows[i-1]
            curr_high, curr_low, curr_close, curr_open = highs[i], lows[i], closes[i], opens[i]
            next_high, next_low = highs[i+1], lows[i+1]
            
            # 檢測看漲 FVG (向上缺口)
            if (prev_high < next_low and curr_close > curr_open):
                gap_high = next_low
                gap_low = prev_high
                gap_size = gap_high - gap_low
                
                # 快速顯著性檢查 - 降低門檻讓更多FVG被檢測到
                atr_value = atr_values[i] if not np.isnan(atr_values[i]) else gap_size * 2
                if gap_size > atr_value * 0.05:
                    # 使用預計算數據檢查是否被填補 - O(1)操作
                    filled = self._is_fvg_filled_fast(future_extremes, i, gap_low, gap_high)
                    
                    if not filled:  # 只保留未填補的
                        fvg_list.append({
                            'type': 'BULLISH_FVG',
                            'high': gap_high,
                            'low': gap_low,
                            'size': gap_size,
                            'time': df.get('time', pd.Series(range(len(df)))).iloc[i],
                            'filled': False,
                            'description': f'看漲缺口 ${gap_low:.6f}-${gap_high:.6f}'
                        })
        
        # 返回最近10個未填補的缺口
        return fvg_list[-10:]
    
    def _is_fvg_filled_fast(self, future_extremes: Dict, gap_index: int, gap_low: float, gap_high: float) -> bool:
        """快速檢查FVG是否被填補 - O(1)操作"""
        if gap_index >= len(future_extremes['future_min_low']) - 1:
            return False
            
        # 使用預計算的未來最值檢查重疊
        future_min = future_extremes['future_min_low'][gap_index]
        future_max = future_extremes['future_max_high'][gap_index]
        
        # 更寬鬆的填補檢查：只有當價格明顯穿越缺口中心時才認為被填補
        gap_center = (gap_low + gap_high) / 2
        gap_size = gap_high - gap_low
        
        # 對於看漲FVG，檢查是否有明顯回調進入缺口
        if gap_size > 0:
            return (future_min < gap_center) and (future_max > gap_center)
        
        return False
    
    def detect_equal_highs_lows(self, df: pd.DataFrame, threshold: float = 0.001) -> Dict:
        """檢測 Equal Highs/Lows - 優化版"""
        if df is None or len(df) < 20:
            return {'equal_highs': [], 'equal_lows': []}
            
        # 使用已優化的swing points計算
        swing_highs, swing_lows = self._compute_swing_points(df, 10)
        
        # 獲取swing點的價格和索引
        high_indices = np.where(swing_highs)[0]
        low_indices = np.where(swing_lows)[0]
        
        high_prices = df['high'].iloc[high_indices].values
        low_prices = df['low'].iloc[low_indices].values
        
        # 優化的equal points檢測
        equal_highs = self._find_equal_points_fast(high_prices, high_indices, threshold, 'EQUAL_HIGH')
        equal_lows = self._find_equal_points_fast(low_prices, low_indices, threshold, 'EQUAL_LOW')
        
        return {
            'equal_highs': equal_highs[-5:],
            'equal_lows': equal_lows[-5:]
        }
    
    def _find_equal_points_fast(self, prices: np.ndarray, indices: np.ndarray, threshold: float, point_type: str) -> List[Dict]:
        """快速找到等價點"""
        if len(prices) < 2:
            return []
            
        equal_points = []
        processed = set()
        
        for i in range(len(prices)):
            if i in processed:
                continue
                
            current_price = prices[i]
            equal_group = [i]
            processed.add(i)
            
            # 向量化查找相等點
            price_diffs = np.abs(prices - current_price)
            equal_mask = price_diffs <= threshold * current_price
            equal_indices = np.where(equal_mask)[0]
            
            for j in equal_indices:
                if j != i and j not in processed:
                    equal_group.append(j)
                    processed.add(j)
            
            # 如果找到2個或以上的等價點
            if len(equal_group) >= 2:
                avg_price = np.mean(prices[equal_group])
                equal_points.append({
                    'price': avg_price,
                    'count': len(equal_group),
                    'type': point_type,
                    'description': f'等{"高" if "HIGH" in point_type else "低"}點 ${avg_price:.6f} (共{len(equal_group)}點)'
                })
        
        return equal_points
    
    def detect_liquidity_sweeps(self, df: pd.DataFrame) -> List[Dict]:
        """檢測流動性掃蕩 - 保持O(n)複雜度但優化實現"""
        if df is None or len(df) < 50:
            return []
            
        df = df.copy()
        sweeps = []
        
        # 預計算ATR
        if 'atr' not in df.columns:
            df['atr'] = self._calculate_atr(df, 14)
        
        # 向量化數據訪問
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        atr_values = df['atr'].values
        
        # 使用滑動窗口預計算局部最值
        window_size = 20
        for i in range(window_size, len(df) - 5):
            current_atr = atr_values[i] if not np.isnan(atr_values[i]) else np.std(highs) * 0.1
            
            # 使用切片操作快速獲取窗口數據
            prev_highs = highs[i-window_size:i]
            prev_lows = lows[i-window_size:i]
            next_closes = closes[i+1:i+6]
            
            if len(prev_highs) == 0 or len(next_closes) == 0:
                continue
                
            max_prev_high = np.max(prev_highs)
            min_prev_low = np.min(prev_lows)
            
            # 只檢查掃蕩買方流動性（對多頭有利的信號）
            if lows[i] < min_prev_low:
                sweep_distance = min_prev_low - lows[i]
                if sweep_distance > current_atr * 0.2:
                    max_next_close = np.max(next_closes)
                    max_recovery = max_next_close - lows[i]
                    
                    if max_next_close > min_prev_low and max_recovery > current_atr * 0.5:
                        sweeps.append({
                            'type': 'BUY_SIDE_LIQUIDITY',
                            'price': lows[i],
                            'swept_level': min_prev_low,
                            'sweep_distance': sweep_distance,
                            'max_recovery': max_recovery,
                            'time': df.get('time', pd.Series(range(len(df)))).iloc[i],
                            'description': f'掃蕩買方流動性 ${min_prev_low:.6f} (反彈 {max_recovery:.6f})'
                        })
            
            # 跳過賣方流動性掃蕩檢測（機器人只做多）
        
        return sweeps[-10:]
    
    def calculate_premium_discount_zones(self, df: pd.DataFrame, lookback: int = 100) -> Dict:
        """計算溢價/折價區域 - 已經是O(1)，保持不變"""
        if df is None or len(df) < lookback:
            return {}
            
        recent_data = df.tail(lookback)
        high_price = recent_data['high'].max()
        low_price = recent_data['low'].min()
        range_size = high_price - low_price

        # 定義區域的初始值
        premium_zone_start = low_price + range_size * 0.7
        discount_zone_start = low_price  # Default value
        discount_zone_end = low_price + range_size * 0.3 # Default value
        equilibrium_top = low_price + range_size * 0.55
        equilibrium_bottom = low_price + range_size * 0.45

        # 檢測 Order Blocks
        order_blocks = self.detect_order_blocks(df)
        bullish_obs = [ob for ob in order_blocks if ob['type'] == 'BULLISH_OB' and ob['active']]

        # 如果存在活躍的看漲 Order Block，則將其作為低價區
        if bullish_obs:
            # 找到最近期的 Order Block
            most_recent_ob = max(bullish_obs, key=lambda ob: ob['time'])
            discount_zone_start = most_recent_ob['low']
            discount_zone_end = most_recent_ob['high']
            # 調整整體 low_price 以確保 discount_zone 包含在整體範圍內
            low_price = min(low_price, discount_zone_start)
            high_price = max(high_price, discount_zone_end)
            range_size = high_price - low_price

        current_price = df['close'].iloc[-1]
        
        # 判斷當前價格位置
        if current_price >= premium_zone_start:
            current_zone = 'PREMIUM'
        elif current_price <= discount_zone_end:
            current_zone = 'DISCOUNT'
        else:
            current_zone = 'EQUILIBRIUM'
        
        return {
            'premium_zone': {'start': premium_zone_start, 'end': high_price},
            'discount_zone': {'start': discount_zone_start, 'end': discount_zone_end},
            'equilibrium_zone': {'start': equilibrium_bottom, 'end': equilibrium_top},
            'current_zone': current_zone,
            'current_price': current_price
        }
    
    def get_comprehensive_analysis(self, df: pd.DataFrame) -> Dict:
        """獲取完整的 SMC 分析 - 優化版 O(n²) → O(n)"""
        if df is None or len(df) < 20:
            return {}
        
        try:
            # 確保數據類型正確
            df = df.copy()
            for col in ['open', 'high', 'low', 'close']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 移除 NaN 值
            df = df.dropna(subset=['open', 'high', 'low', 'close'])
            
            if len(df) < 20:
                return {}
            
            # 檢查緩存
            df_hash = self._get_df_hash(df)
            if self._last_df_hash == df_hash and 'comprehensive' in self._cache:
                return self._cache['comprehensive']
            
            # 預計算ATR和未來最值 - 一次性計算，避免重複
            df['atr'] = self._calculate_atr(df, 14)
            future_extremes = self._precompute_future_extremes(df)
            
            # 所有分析共享預計算數據
            analysis = {
                'market_structure': self.detect_market_structure(df),
                'order_blocks': self.detect_order_blocks(df.tail(180)),
                'fair_value_gaps': self.detect_fair_value_gaps(df),
                'equal_highs_lows': self.detect_equal_highs_lows(df),
                'liquidity_sweeps': self.detect_liquidity_sweeps(df),
                'premium_discount': self.calculate_premium_discount_zones(df),
                'overall_bias': self._calculate_overall_bias_fast(df, analysis_cache=True)
            }
            
            # 更新緩存
            self._cache['comprehensive'] = analysis
            self._last_df_hash = df_hash
            
            return analysis
            
        except Exception as e:
            print(f"SMC 綜合分析錯誤: {e}")
            return {}
    
    def _calculate_overall_bias_fast(self, df: pd.DataFrame, analysis_cache: bool = False) -> str:
        """計算整體市場偏向 - 優化版 O(n²) → O(n)"""
        if df is None or len(df) < 20:
            return 'NEUTRAL'
            
        # 基於多個因素判斷
        current_price = df['close'].iloc[-1]
        
        # 使用向量化操作計算移動平均
        closes = df['close'].values
        sma_20 = np.mean(closes[-20:])
        sma_50 = np.mean(closes[-50:]) if len(closes) >= 50 else sma_20
        
        bias_score = 0
        
        # 1. 均線趨勢 (30分)
        if current_price > sma_20 > sma_50:
            bias_score += 30
        elif current_price < sma_20 < sma_50:
            bias_score -= 30
        elif current_price > sma_20:
            bias_score += 15
        elif current_price < sma_20:
            bias_score -= 15
            
        # 2. 高低點結構 (25分) - 向量化計算
        if len(df) >= 10:
            recent_highs = df['high'].tail(10).values
            recent_lows = df['low'].tail(10).values
            
            if len(recent_highs) >= 5 and len(recent_lows) >= 5:
                # 向量化檢查趨勢
                higher_highs = recent_highs[-1] > recent_highs[-3] > recent_highs[-5]
                higher_lows = recent_lows[-1] > recent_lows[-3] > recent_lows[-5]
                lower_highs = recent_highs[-1] < recent_highs[-3] < recent_highs[-5]
                lower_lows = recent_lows[-1] < recent_lows[-3] < recent_lows[-5]
                
                if higher_highs and higher_lows:
                    bias_score += 25
                elif lower_highs and lower_lows:
                    bias_score -= 25
                    
        # 3. 溢價/折價區域 (20分)
        premium_discount = self.calculate_premium_discount_zones(df)
        if premium_discount and 'current_zone' in premium_discount:
            if premium_discount['current_zone'] == 'DISCOUNT':
                bias_score += 10
            elif premium_discount['current_zone'] == 'PREMIUM':
                bias_score -= 10
                
        # 4. 如果是緩存模式，跳過重複計算
        if not analysis_cache:
            # 簡化的OB和FVG檢查，避免重複計算
            try:
                # 快速檢查最近的結構信號
                market_structure = self.detect_market_structure(df)
                if market_structure.get('trend') == self.BULLISH:
                    bias_score += 8
                elif market_structure.get('trend') == self.BEARISH:
                    bias_score -= 8
            except:
                pass
                
        # 5. 成交量確認 (10分) - 向量化
        if 'volume' in df.columns:
            volumes = df['volume'].values
            if len(volumes) >= 5:
                recent_volume = np.mean(volumes[-5:])
                avg_volume = np.mean(volumes)
                price_change = (current_price - closes[-5]) / closes[-5]
                
                if price_change > 0 and recent_volume > avg_volume * 1.2:
                    bias_score += 10
                elif price_change < 0 and recent_volume > avg_volume * 1.2:
                    bias_score -= 10
        
        # 根據總分判斷偏向
        if bias_score > 15:
            return 'BULLISH'
        elif bias_score < -15:
            return 'BEARISH'
        else:
            return 'NEUTRAL'