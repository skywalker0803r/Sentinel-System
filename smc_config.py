"""
Smart Money Concepts 配置文件
可以調整各種分析參數和評分權重
"""

class SMCConfig:
    """SMC 分析配置"""
    
    # ====== 市場結構檢測參數 ======
    MARKET_STRUCTURE = {
        'lookback_period': 50,              # 回看週期
        'min_swing_strength': 10,           # 最小 swing 強度
        'structure_confirmation_bars': 3,   # 結構確認K線數
    }
    
    # ====== Order Blocks 參數 ======
    ORDER_BLOCKS = {
        'lookback_period': 100,             # 回看週期
        'atr_multiplier': 0.5,              # ATR 過濾倍數
        'max_blocks_per_type': 10,          # 每種類型最大保留數量
        'min_strength_threshold': 30,       # 最小強度閾值
    }
    
    # ====== Fair Value Gaps 參數 ======
    FAIR_VALUE_GAPS = {
        'atr_multiplier': 0.3,              # ATR 過濾倍數
        'max_gaps_to_track': 10,            # 最大追蹤缺口數
        'require_volume_confirmation': False, # 是否需要成交量確認
    }
    
    # ====== Equal Highs/Lows 參數 ======
    EQUAL_HIGHS_LOWS = {
        'price_threshold': 0.001,           # 價格相似度閾值 (0.1%)
        'min_bars_apart': 10,               # 最小間隔K線數
        'max_levels_to_track': 5,           # 最大追蹤水平數
    }
    
    # ====== 流動性掃蕩參數 ======
    LIQUIDITY_SWEEPS = {
        'lookback_bars': 20,                # 回看K線數
        'confirmation_bars': 5,             # 確認K線數
        'min_sweep_distance': 0.002,        # 最小掃蕩距離 (0.2%)
    }
    
    # ====== Premium/Discount 區域 ======
    PREMIUM_DISCOUNT = {
        'lookback_period': 100,             # 回看週期
        'premium_threshold': 0.7,           # 溢價區域閾值 (70%)
        'discount_threshold': 0.3,          # 折價區域閾值 (30%)
        'equilibrium_range': 0.1,           # 平衡區域範圍 (±5%)
    }

class ScoringConfig:
    """評分系統配置"""
    
    # ====== 評分權重 ======
    SCORE_WEIGHTS = {
        # Vegas 通道基礎分數
        'vegas_breakout': 25,               # 突破訊號
        'vegas_bounce': 15,                 # 反彈訊號
        
        # SMC 市場結構分數
        'smc_bos': 15,                      # BOS 確認
        'smc_choch': 20,                    # CHoCH 轉勢
        
        # Order Blocks 分數
        'order_blocks_max': 15,             # Order Blocks 最高分
        'order_blocks_per_block': 3,        # 每個活躍 OB 得分
        
        # Fair Value Gaps 分數
        'fvg_max': 10,                      # FVG 最高分
        'fvg_per_gap': 2,                   # 每個 FVG 得分
        
        # 流動性掃蕩分數
        'liquidity_max': 10,                # 流動性掃蕩最高分
        'liquidity_per_sweep': 2,           # 每個掃蕩得分
        
        # 年利率加成分數
        'apr_high': 10,                     # 100%+ APR
        'apr_medium': 6,                    # 50%+ APR
        'apr_low': 3,                       # 20%+ APR
    }
    
    # ====== 分層閾值 ======
    TIER_THRESHOLDS = {
        'tier1_min': 70,                    # Tier 1 最低分數
        'tier2_min': 50,                    # Tier 2 最低分數
        'tier3_min': 30,                    # Tier 3 最低分數
    }
    
    # ====== 顯示數量限制 ======
    DISPLAY_LIMITS = {
        'tier1_count': 3,                   # Tier 1 顯示數量
        'tier2_count': 5,                   # Tier 2 顯示數量
        'tier3_count': 5,                   # Tier 3 顯示數量
        'max_symbols_scan': 200,            # 最大掃描交易對數
    }

class DiscordConfig:
    """Discord 顯示配置"""
    
    # ====== 訊號 Emoji ======
    SIGNAL_EMOJIS = {
        'LONG_BREAKOUT': '🚀',
        'LONG_BOUNCE': '⬆️',
        'SHORT_BREAKDOWN': '📉',
        'SHORT_FAILED_BOUNCE': '⬇️',
        'SMC_BULLISH': '🔥',
        'SMC_BEARISH': '❄️',
        'UNKNOWN': '❓'
    }
    
    # ====== 區域 Emoji ======
    ZONE_EMOJIS = {
        'PREMIUM': '🔴',
        'DISCOUNT': '🟢',
        'EQUILIBRIUM': '🟡'
    }
    
    # ====== Embed 顏色 ======
    EMBED_COLORS = {
        'analyzing': 0xFFD700,              # 金色 - 分析中
        'success': 0x00FF00,                # 綠色 - 成功
        'no_signals': 0x808080,             # 灰色 - 無訊號
        'error': 0xFF0000,                  # 紅色 - 錯誤
    }
    
    # ====== 訊息文本 ======
    MESSAGES = {
        'analyzing_title': '🔍 增強版技術分析中...',
        'analyzing_desc': '正在掃描所有交易對並計算 Vegas 通道 + Smart Money Concepts 指標，請稍候...',
        'no_signals_title': '📊 技術分析結果',
        'no_signals_desc': '目前沒有符合條件的交易訊號。',
        'main_title': '🎯 增強版技術分析報告',
        'main_desc': '結合 Vegas 通道 + Smart Money Concepts 的綜合分析',
        'footer_text': '⚠️ 僅供參考，請自行評估風險 | 結合多重技術指標分析'
    }

class TradingConfig:
    """交易相關配置"""
    
    # ====== 風險管理 ======
    RISK_MANAGEMENT = {
        'max_signals_per_run': 50,          # 每次最大訊號數
        'min_volume_threshold': 1000,       # 最小成交量閾值
        'blacklist_symbols': [],            # 黑名單交易對
        'preferred_quote_currencies': ['USDT', 'BTC', 'ETH'],  # 偏好計價幣種
    }
    
    # ====== APR 閾值 ======
    APR_THRESHOLDS = {
        'high': 1.0,                        # 100%+ 高年利率
        'medium': 0.5,                      # 50%+ 中年利率
        'low': 0.2,                         # 20%+ 低年利率
        'minimum': 0.05,                    # 5%+ 最低年利率
    }

# ====== 快速配置函數 ======

def get_conservative_config():
    """保守型配置 - 更高的閾值，更少的訊號"""
    config = SMCConfig()
    config.ORDER_BLOCKS['min_strength_threshold'] = 50
    config.FAIR_VALUE_GAPS['atr_multiplier'] = 0.5
    
    scoring = ScoringConfig()
    scoring.TIER_THRESHOLDS['tier1_min'] = 80
    scoring.TIER_THRESHOLDS['tier2_min'] = 60
    
    return config, scoring

def get_aggressive_config():
    """積極型配置 - 更低的閾值，更多的訊號"""
    config = SMCConfig()
    config.ORDER_BLOCKS['min_strength_threshold'] = 20
    config.FAIR_VALUE_GAPS['atr_multiplier'] = 0.2
    
    scoring = ScoringConfig()
    scoring.TIER_THRESHOLDS['tier1_min'] = 60
    scoring.TIER_THRESHOLDS['tier2_min'] = 40
    scoring.DISPLAY_LIMITS['tier1_count'] = 5
    
    return config, scoring

def get_balanced_config():
    """平衡型配置 - 默認設置"""
    return SMCConfig(), ScoringConfig()

# ====== 配置驗證 ======

def validate_config(smc_config, scoring_config):
    """驗證配置的合理性"""
    errors = []
    
    # 檢查評分總和
    max_possible_score = (
        max(scoring_config.SCORE_WEIGHTS['vegas_breakout'], scoring_config.SCORE_WEIGHTS['vegas_bounce']) +
        max(scoring_config.SCORE_WEIGHTS['smc_bos'], scoring_config.SCORE_WEIGHTS['smc_choch']) +
        scoring_config.SCORE_WEIGHTS['order_blocks_max'] +
        scoring_config.SCORE_WEIGHTS['fvg_max'] +
        scoring_config.SCORE_WEIGHTS['liquidity_max'] +
        scoring_config.SCORE_WEIGHTS['apr_high']
    )
    
    if max_possible_score > 100:
        errors.append(f"最大可能分數 ({max_possible_score}) 超過 100")
    
    # 檢查分層閾值
    tiers = scoring_config.TIER_THRESHOLDS
    if not (tiers['tier3_min'] < tiers['tier2_min'] < tiers['tier1_min']):
        errors.append("分層閾值順序不正確")
    
    # 檢查顯示限制
    if scoring_config.DISPLAY_LIMITS['max_symbols_scan'] < 10:
        errors.append("掃描交易對數量過少")
    
    return errors

# ====== 使用示例 ======
if __name__ == "__main__":
    # 獲取平衡配置
    smc_config, scoring_config = get_balanced_config()
    
    # 驗證配置
    errors = validate_config(smc_config, scoring_config)
    
    if errors:
        print("❌ 配置錯誤:")
        for error in errors:
            print(f"   - {error}")
    else:
        print("✅ 配置驗證通過")
        print(f"📊 最大可能評分: {sum(scoring_config.SCORE_WEIGHTS.values())}")
        print(f"🎯 分層閾值: T1≥{scoring_config.TIER_THRESHOLDS['tier1_min']}, T2≥{scoring_config.TIER_THRESHOLDS['tier2_min']}, T3≥{scoring_config.TIER_THRESHOLDS['tier3_min']}")