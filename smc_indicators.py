import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

class SmartMoneyConceptsAnalyzer:
    """Smart Money Concepts 技術分析器"""
    
    def __init__(self):
        self.BULLISH = 1
        self.BEARISH = -1
        
    def detect_market_structure(self, df: pd.DataFrame, lookback: int = 50) -> Dict:
        """檢測市場結構：BOS (Break of Structure) 和 CHoCH (Change of Character)"""
        if df is None or len(df) < lookback:
            return {'bos_signals': [], 'choch_signals': [], 'trend': None}
            
        df = df.copy()
        
        # 計算 ATR 用於過濾顯著性
        if 'atr' not in df.columns:
            df['atr'] = self._calculate_atr(df, 14)
        
        # 計算 swing highs 和 swing lows (加入顯著性過濾)
        window = lookback//2
        df['swing_high'] = df['high'].rolling(window=window, center=True).max() == df['high']
        df['swing_low'] = df['low'].rolling(window=window, center=True).min() == df['low']
        
        # 過濾不顯著的 swing points (距離小於 ATR * 0.5)
        current_atr = df['atr'].iloc[-1] if not pd.isna(df['atr'].iloc[-1]) else df['high'].std() * 0.1
        min_swing_distance = current_atr * 0.5
        
        # 識別重要的 swing points
        swing_highs = df[df['swing_high']]['high'].dropna()
        swing_lows = df[df['swing_low']]['low'].dropna()
        
        bos_signals = []
        choch_signals = []
        current_trend = None
        
        # 檢測結構突破
        current_price = df['close'].iloc[-1]
        
        if len(swing_highs) >= 2:
            last_high = swing_highs.iloc[-1]
            prev_high = swing_highs.iloc[-2]
            
            # BOS: 突破前高 (上升趨勢延續)
            if current_price > last_high and last_high > prev_high:
                bos_signals.append({
                    'type': 'BOS_BULLISH',
                    'price': current_price,
                    'strength': self._calculate_structure_strength(df, 'bullish'),
                    'description': f'突破前高 ${last_high:.6f}'
                })
                current_trend = self.BULLISH
                
            # CHoCH: 突破前高後又跌破 (趨勢轉變)
            elif current_price < prev_high and last_high > prev_high:
                choch_signals.append({
                    'type': 'CHOCH_BEARISH',
                    'price': current_price,
                    'strength': self._calculate_structure_strength(df, 'bearish'),
                    'description': f'跌破前高支撐 ${prev_high:.6f}'
                })
                current_trend = self.BEARISH
        
        if len(swing_lows) >= 2:
            last_low = swing_lows.iloc[-1]
            prev_low = swing_lows.iloc[-2]
            
            # BOS: 跌破前低 (下降趨勢延續)
            if current_price < last_low and last_low < prev_low:
                bos_signals.append({
                    'type': 'BOS_BEARISH',
                    'price': current_price,
                    'strength': self._calculate_structure_strength(df, 'bearish'),
                    'description': f'跌破前低 ${last_low:.6f}'
                })
                current_trend = self.BEARISH
                
            # CHoCH: 跌破前低後又漲破 (趨勢轉變)
            elif current_price > prev_low and last_low < prev_low:
                choch_signals.append({
                    'type': 'CHOCH_BULLISH',
                    'price': current_price,
                    'strength': self._calculate_structure_strength(df, 'bullish'),
                    'description': f'突破前低阻力 ${prev_low:.6f}'
                })
                current_trend = self.BULLISH
        
        return {
            'bos_signals': bos_signals,
            'choch_signals': choch_signals,
            'trend': current_trend,
            'swing_highs': swing_highs.tolist()[-5:] if len(swing_highs) > 0 else [],
            'swing_lows': swing_lows.tolist()[-5:] if len(swing_lows) > 0 else []
        }
    
    def detect_order_blocks(self, df: pd.DataFrame, lookback: int = 100) -> List[Dict]:
        """檢測 Order Blocks (機構訂單區塊)"""
        if df is None or len(df) < lookback:
            return []
            
        df = df.copy()
        order_blocks = []
        
        # 計算 ATR 用於過濾
        df['atr'] = self._calculate_atr(df, 14)
        current_atr = df['atr'].iloc[-1]
        
        # 尋找顯著的反轉蠟燭
        for i in range(lookback, len(df) - 5):
            current_candle = df.iloc[i]
            
            # 檢查是否為反轉點
            is_bullish_ob = self._is_bullish_order_block(df, i)
            is_bearish_ob = self._is_bearish_order_block(df, i)
            
            if is_bullish_ob:
                ob_high = current_candle['high']
                ob_low = current_candle['low']
                
                # 檢查是否足夠顯著 (大於平均 ATR)
                if (ob_high - ob_low) > current_atr * 0.5:
                    order_blocks.append({
                        'type': 'BULLISH_OB',
                        'high': ob_high,
                        'low': ob_low,
                        'time': current_candle.get('time', i),
                        'strength': self._calculate_ob_strength(df, i, 'bullish'),
                        'active': self._is_order_block_active(df, ob_low, ob_high, i, 'BULLISH_OB'),
                        'description': f'看漲訂單區塊 ${ob_low:.6f}-${ob_high:.6f}'
                    })
            
            if is_bearish_ob:
                ob_high = current_candle['high']
                ob_low = current_candle['low']
                
                if (ob_high - ob_low) > current_atr * 0.5:
                    order_blocks.append({
                        'type': 'BEARISH_OB',
                        'high': ob_high,
                        'low': ob_low,
                        'time': current_candle.get('time', i),
                        'strength': self._calculate_ob_strength(df, i, 'bearish'),
                        'active': self._is_order_block_active(df, ob_low, ob_high, i, 'BEARISH_OB'),
                        'description': f'看跌訂單區塊 ${ob_low:.6f}-${ob_high:.6f}'
                    })
        
        # 只返回最近的活躍 Order Blocks
        active_obs = [ob for ob in order_blocks if ob['active']]
        return sorted(active_obs, key=lambda x: x['strength'], reverse=True)[:10]
    
    def detect_fair_value_gaps(self, df: pd.DataFrame) -> List[Dict]:
        """檢測 Fair Value Gaps (公允價值缺口)"""
        if df is None or len(df) < 3:
            return []
            
        df = df.copy()
        fvg_list = []
        
        for i in range(1, len(df) - 1):
            prev_candle = df.iloc[i-1]
            current_candle = df.iloc[i]
            next_candle = df.iloc[i+1]
            
            # 檢測看漲 FVG (向上缺口)
            if (prev_candle['high'] < next_candle['low'] and 
                current_candle['close'] > current_candle['open']):
                
                gap_high = next_candle['low']
                gap_low = prev_candle['high']
                gap_size = gap_high - gap_low
                
                # 檢查缺口是否顯著 (降低閾值以檢測更多有效缺口)
                atr_value = df['atr'].iloc[i] if 'atr' in df.columns else gap_size * 2
                if gap_size > atr_value * 0.1:
                    fvg_list.append({
                        'type': 'BULLISH_FVG',
                        'high': gap_high,
                        'low': gap_low,
                        'size': gap_size,
                        'time': current_candle.get('time', i),
                        'filled': self._is_fvg_filled(df, gap_low, gap_high, i),
                        'description': f'看漲缺口 ${gap_low:.6f}-${gap_high:.6f}'
                    })
            
            # 檢測看跌 FVG (向下缺口)
            elif (prev_candle['low'] > next_candle['high'] and 
                  current_candle['close'] < current_candle['open']):
                
                gap_high = prev_candle['low']
                gap_low = next_candle['high']
                gap_size = gap_high - gap_low
                
                atr_value = df['atr'].iloc[i] if 'atr' in df.columns else gap_size * 2
                if gap_size > atr_value * 0.1:
                    fvg_list.append({
                        'type': 'BEARISH_FVG',
                        'high': gap_high,
                        'low': gap_low,
                        'size': gap_size,
                        'time': current_candle.get('time', i),
                        'filled': self._is_fvg_filled(df, gap_low, gap_high, i),
                        'description': f'看跌缺口 ${gap_low:.6f}-${gap_high:.6f}'
                    })
        
        # 只返回未填補的缺口
        unfilled_fvgs = [fvg for fvg in fvg_list if not fvg['filled']]
        return unfilled_fvgs[-10:]  # 最近10個
    
    def detect_equal_highs_lows(self, df: pd.DataFrame, threshold: float = 0.001) -> Dict:
        """檢測 Equal Highs/Lows (等高點/等低點)"""
        if df is None or len(df) < 20:
            return {'equal_highs': [], 'equal_lows': []}
            
        df = df.copy()
        
        # 找出 swing points
        swing_highs = []
        swing_lows = []
        
        for i in range(10, len(df) - 10):
            # 檢查是否為 swing high
            if (df['high'].iloc[i] == df['high'].iloc[i-10:i+11].max()):
                swing_highs.append({
                    'price': df['high'].iloc[i],
                    'index': i,
                    'time': df.get('time', pd.Series([i] * len(df))).iloc[i]
                })
            
            # 檢查是否為 swing low
            if (df['low'].iloc[i] == df['low'].iloc[i-10:i+11].min()):
                swing_lows.append({
                    'price': df['low'].iloc[i],
                    'index': i,
                    'time': df.get('time', pd.Series([i] * len(df))).iloc[i]
                })
        
        # 檢測 equal highs (支援 3+ 點統計)
        equal_highs = []
        processed_highs = set()
        
        for i in range(len(swing_highs)):
            if i in processed_highs:
                continue
                
            equal_group = [swing_highs[i]]
            processed_highs.add(i)
            
            # 尋找所有與當前點相等的點
            for j in range(i + 1, len(swing_highs)):
                if j in processed_highs:
                    continue
                    
                price_diff = abs(swing_highs[i]['price'] - swing_highs[j]['price'])
                if price_diff <= threshold * swing_highs[i]['price']:
                    equal_group.append(swing_highs[j])
                    processed_highs.add(j)
            
            # 如果找到2個或以上的等高點
            if len(equal_group) >= 2:
                avg_price = sum(point['price'] for point in equal_group) / len(equal_group)
                equal_highs.append({
                    'price': avg_price,
                    'count': len(equal_group),
                    'type': 'EQUAL_HIGH',
                    'description': f'等高點 ${avg_price:.6f} (共{len(equal_group)}點)'
                })
        
        # 檢測 equal lows (支援 3+ 點統計)
        equal_lows = []
        processed_lows = set()
        
        for i in range(len(swing_lows)):
            if i in processed_lows:
                continue
                
            equal_group = [swing_lows[i]]
            processed_lows.add(i)
            
            # 尋找所有與當前點相等的點
            for j in range(i + 1, len(swing_lows)):
                if j in processed_lows:
                    continue
                    
                price_diff = abs(swing_lows[i]['price'] - swing_lows[j]['price'])
                if price_diff <= threshold * swing_lows[i]['price']:
                    equal_group.append(swing_lows[j])
                    processed_lows.add(j)
            
            # 如果找到2個或以上的等低點
            if len(equal_group) >= 2:
                avg_price = sum(point['price'] for point in equal_group) / len(equal_group)
                equal_lows.append({
                    'price': avg_price,
                    'count': len(equal_group),
                    'type': 'EQUAL_LOW',
                    'description': f'等低點 ${avg_price:.6f} (共{len(equal_group)}點)'
                })
        
        return {
            'equal_highs': equal_highs[-5:],  # 最近5個
            'equal_lows': equal_lows[-5:]
        }
    
    def detect_liquidity_sweeps(self, df: pd.DataFrame) -> List[Dict]:
        """檢測流動性掃蕩 (Liquidity Sweeps) - 增強版本"""
        if df is None or len(df) < 50:
            return []
            
        df = df.copy()
        sweeps = []
        
        # 計算 ATR 用於過濾顯著性
        if 'atr' not in df.columns:
            df['atr'] = self._calculate_atr(df, 14)
        
        # 尋找顯著的高點和低點
        for i in range(20, len(df) - 5):
            current_high = df['high'].iloc[i]
            current_low = df['low'].iloc[i]
            current_atr = df['atr'].iloc[i] if not pd.isna(df['atr'].iloc[i]) else df['high'].std() * 0.1
            
            # 檢查是否掃蕩了之前的高點 (賣方流動性)
            prev_highs = df['high'].iloc[i-20:i]
            if len(prev_highs) > 0:
                max_prev_high = prev_highs.max()
                sweep_distance = current_high - max_prev_high
                
                # 掃蕩距離必須顯著 (大於 ATR * 0.2) 且實際突破
                if current_high > max_prev_high and sweep_distance > current_atr * 0.2:
                    # 檢查後續是否快速回落 (假突破)
                    next_candles = df.iloc[i+1:i+6]
                    if len(next_candles) > 0:
                        min_next_close = next_candles['close'].min()
                        max_drawdown = current_high - min_next_close
                        
                        # 回撤必須顯著 (大於 ATR * 0.5) 且跌破掃蕩水平
                        if min_next_close < max_prev_high and max_drawdown > current_atr * 0.5:
                            sweeps.append({
                                'type': 'SELL_SIDE_LIQUIDITY',
                                'price': current_high,
                                'swept_level': max_prev_high,
                                'sweep_distance': sweep_distance,
                                'max_drawdown': max_drawdown,
                                'time': df.get('time', pd.Series([i] * len(df))).iloc[i],
                                'description': f'掃蕩賣方流動性 ${max_prev_high:.6f} (回撤 {max_drawdown:.6f})'
                            })
            
            # 檢查是否掃蕩了之前的低點 (買方流動性)
            prev_lows = df['low'].iloc[i-20:i]
            if len(prev_lows) > 0:
                min_prev_low = prev_lows.min()
                sweep_distance = min_prev_low - current_low
                
                # 掃蕩距離必須顯著且實際跌破
                if current_low < min_prev_low and sweep_distance > current_atr * 0.2:
                    # 檢查後續是否快速反彈 (假跌破)
                    next_candles = df.iloc[i+1:i+6]
                    if len(next_candles) > 0:
                        max_next_close = next_candles['close'].max()
                        max_recovery = max_next_close - current_low
                        
                        # 反彈必須顯著且突破掃蕩水平
                        if max_next_close > min_prev_low and max_recovery > current_atr * 0.5:
                            sweeps.append({
                                'type': 'BUY_SIDE_LIQUIDITY',
                                'price': current_low,
                                'swept_level': min_prev_low,
                                'sweep_distance': sweep_distance,
                                'max_recovery': max_recovery,
                                'time': df.get('time', pd.Series([i] * len(df))).iloc[i],
                                'description': f'掃蕩買方流動性 ${min_prev_low:.6f} (反彈 {max_recovery:.6f})'
                            })
        
        return sweeps[-10:]  # 最近10個
    
    def calculate_premium_discount_zones(self, df: pd.DataFrame, lookback: int = 100) -> Dict:
        """計算溢價/折價區域"""
        if df is None or len(df) < lookback:
            return {}
            
        recent_data = df.tail(lookback)
        high_price = recent_data['high'].max()
        low_price = recent_data['low'].min()
        range_size = high_price - low_price
        
        # 定義區域
        premium_zone_start = low_price + range_size * 0.7
        discount_zone_end = low_price + range_size * 0.3
        equilibrium_top = low_price + range_size * 0.55
        equilibrium_bottom = low_price + range_size * 0.45
        
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
            'discount_zone': {'start': low_price, 'end': discount_zone_end},
            'equilibrium_zone': {'start': equilibrium_bottom, 'end': equilibrium_top},
            'current_zone': current_zone,
            'current_price': current_price
        }
    
    def get_comprehensive_analysis(self, df: pd.DataFrame) -> Dict:
        """獲取完整的 SMC 分析"""
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
            
            # 計算 ATR
            df['atr'] = self._calculate_atr(df, 14)
            
            analysis = {
                'market_structure': self.detect_market_structure(df),
                'order_blocks': self.detect_order_blocks(df),
                'fair_value_gaps': self.detect_fair_value_gaps(df),
                'equal_highs_lows': self.detect_equal_highs_lows(df),
                'liquidity_sweeps': self.detect_liquidity_sweeps(df),
                'premium_discount': self.calculate_premium_discount_zones(df),
                'overall_bias': self._calculate_overall_bias(df)
            }
            
            return analysis
            
        except Exception as e:
            print(f"SMC 綜合分析錯誤: {e}")
            return {}
    
    # 輔助函數
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """計算 Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()
    
    def _calculate_structure_strength(self, df: pd.DataFrame, direction: str) -> int:
        """計算結構強度 (0-100)"""
        volume_factor = 50  # 基礎分數
        
        # 根據成交量調整 (如果有成交量數據)
        if 'volume' in df.columns:
            recent_volume = df['volume'].tail(10).mean()
            avg_volume = df['volume'].mean()
            if recent_volume > avg_volume * 1.5:
                volume_factor += 20
        
        # 根據 ATR 調整
        current_atr = df['atr'].iloc[-1] if 'atr' in df.columns else 0
        if current_atr > df['atr'].mean() * 1.2 if 'atr' in df.columns else False:
            volume_factor += 15
        
        return min(100, volume_factor)
    
    def _is_bullish_order_block(self, df: pd.DataFrame, index: int) -> bool:
        """檢查是否為看漲 Order Block"""
        if index < 5 or index >= len(df) - 5:
            return False
            
        # 檢查是否有明顯的向上反轉
        current = df.iloc[index]
        prev_candles = df.iloc[index-5:index]
        next_candles = df.iloc[index+1:index+6]
        
        # 前面應該是下跌
        declining = prev_candles['close'].iloc[-1] < prev_candles['close'].iloc[0]
        # 後面應該是上漲
        rising = len(next_candles) > 0 and next_candles['close'].iloc[-1] > current['close']
        
        return declining and rising
    
    def _is_bearish_order_block(self, df: pd.DataFrame, index: int) -> bool:
        """檢查是否為看跌 Order Block"""
        if index < 5 or index >= len(df) - 5:
            return False
            
        current = df.iloc[index]
        prev_candles = df.iloc[index-5:index]
        next_candles = df.iloc[index+1:index+6]
        
        # 前面應該是上漲
        rising = prev_candles['close'].iloc[-1] > prev_candles['close'].iloc[0]
        # 後面應該是下跌
        declining = len(next_candles) > 0 and next_candles['close'].iloc[-1] < current['close']
        
        return rising and declining
    
    def _calculate_ob_strength(self, df: pd.DataFrame, index: int, ob_type: str) -> int:
        """計算 Order Block 強度"""
        base_strength = 50
        
        # 根據成交量增加強度
        if 'volume' in df.columns and index < len(df):
            current_volume = df['volume'].iloc[index]
            avg_volume = df['volume'].mean()
            if current_volume > avg_volume * 1.5:
                base_strength += 25
        
        return min(100, base_strength)
    
    def _is_order_block_active(self, df: pd.DataFrame, ob_low: float, ob_high: float, ob_index: int, ob_type: str = None) -> bool:
        """檢查 Order Block 是否仍然活躍"""
        # 檢查後續價格是否已經完全穿越該區域
        future_data = df.iloc[ob_index+1:]
        if len(future_data) == 0:
            return True
            
        min_future_low = future_data['low'].min()
        max_future_high = future_data['high'].max()
        
        # 根據 OB 類型分別判斷失效條件
        if ob_type == 'BULLISH_OB':
            # 多頭 OB：跌破 ob_low 就失效
            return min_future_low >= ob_low
        elif ob_type == 'BEARISH_OB':
            # 空頭 OB：突破 ob_high 就失效
            return max_future_high <= ob_high
        else:
            # 向後兼容：原有邏輯
            return not (min_future_low < ob_low and max_future_high > ob_high)
    
    def _is_fvg_filled(self, df: pd.DataFrame, gap_low: float, gap_high: float, gap_index: int) -> bool:
        """檢查 Fair Value Gap 是否已被填補 (向量化版本)"""
        future_data = df.iloc[gap_index+1:]
        if len(future_data) == 0:
            return False
            
        # 使用向量化操作檢查後續價格是否重新進入缺口區域
        future = future_data[['low', 'high']]
        filled = ((future['low'] <= gap_high) & (future['high'] >= gap_low)).any()
        return filled
    
    def _calculate_overall_bias(self, df: pd.DataFrame) -> str:
        """計算整體市場偏向 - 增強版本"""
        if df is None or len(df) < 20:
            return 'NEUTRAL'
            
        # 基於多個因素判斷
        current_price = df['close'].iloc[-1]
        sma_20 = df['close'].tail(20).mean()
        sma_50 = df['close'].tail(50).mean() if len(df) >= 50 else sma_20
        
        bias_score = 0
        max_score = 100
        
        # 1. 均線趨勢 (30分)
        if current_price > sma_20 > sma_50:
            bias_score += 30
        elif current_price < sma_20 < sma_50:
            bias_score -= 30
        elif current_price > sma_20:
            bias_score += 15
        elif current_price < sma_20:
            bias_score -= 15
            
        # 2. 高低點結構 (25分)
        recent_highs = df['high'].tail(10)
        recent_lows = df['low'].tail(10)
        
        # 檢查是否形成更高的高點和更高的低點 (上升趨勢)
        if len(recent_highs) >= 5 and len(recent_lows) >= 5:
            higher_highs = recent_highs.iloc[-1] > recent_highs.iloc[-3] > recent_highs.iloc[-5]
            higher_lows = recent_lows.iloc[-1] > recent_lows.iloc[-3] > recent_lows.iloc[-5]
            lower_highs = recent_highs.iloc[-1] < recent_highs.iloc[-3] < recent_highs.iloc[-5]
            lower_lows = recent_lows.iloc[-1] < recent_lows.iloc[-3] < recent_lows.iloc[-5]
            
            if higher_highs and higher_lows:
                bias_score += 25
            elif lower_highs and lower_lows:
                bias_score -= 25
                
        # 3. 溢價/折價區域 (20分)
        premium_discount = self.calculate_premium_discount_zones(df)
        if premium_discount and 'current_zone' in premium_discount:
            if premium_discount['current_zone'] == 'DISCOUNT':
                bias_score += 10  # 折價區偏向看漲
            elif premium_discount['current_zone'] == 'PREMIUM':
                bias_score -= 10  # 溢價區偏向看跌
                
        # 4. 近期Order Blocks 和 FVG 分布 (15分)
        try:
            order_blocks = self.detect_order_blocks(df.copy())
            bullish_obs = len([ob for ob in order_blocks if ob['type'] == 'BULLISH_OB' and ob['active']])
            bearish_obs = len([ob for ob in order_blocks if ob['type'] == 'BEARISH_OB' and ob['active']])
            
            if bullish_obs > bearish_obs:
                bias_score += 8
            elif bearish_obs > bullish_obs:
                bias_score -= 8
                
            fvgs = self.detect_fair_value_gaps(df.copy())
            bullish_fvgs = len([fvg for fvg in fvgs if fvg['type'] == 'BULLISH_FVG' and not fvg['filled']])
            bearish_fvgs = len([fvg for fvg in fvgs if fvg['type'] == 'BEARISH_FVG' and not fvg['filled']])
            
            if bullish_fvgs > bearish_fvgs:
                bias_score += 7
            elif bearish_fvgs > bullish_fvgs:
                bias_score -= 7
        except:
            pass  # 如果計算失敗，跳過這部分評分
            
        # 5. 成交量確認 (10分)
        if 'volume' in df.columns:
            recent_volume = df['volume'].tail(5).mean()
            avg_volume = df['volume'].mean()
            price_change = (current_price - df['close'].iloc[-5]) / df['close'].iloc[-5]
            
            # 上漲配合放量或下跌配合放量
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