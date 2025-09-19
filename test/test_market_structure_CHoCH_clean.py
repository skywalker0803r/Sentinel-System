import sys
import os
sys.path.append(os.path.abspath(".."))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from smc_indicators_optimized import OptimizedSmartMoneyConceptsAnalyzer
from get_real_data import get_btc_data
import warnings
warnings.filterwarnings('ignore')

# ===== 獲取真實比特幣數據用於CHoCH分析 =====
print("=== CHoCH (Change of Character) 分析 - 使用真實BTC數據 ===")
df = get_btc_data('BTC_USDT', '6h', 130)  # 獲取130根6小時K線

print(f"\n真實BTC數據:")
print(df[['time', 'open', 'high', 'low', 'close', 'volume']].head(10))
print("...")
print(df[['time', 'open', 'high', 'low', 'close', 'volume']].tail(10))

print(f"\n=== CHoCH (Change of Character) 測試數據統計 ===")
print(f"數據期間: {df['time'].iloc[0]} 到 {df['time'].iloc[-1]}")
print(f"總K線數: {len(df)}")
print(f"價格範圍: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
print(f"總漲跌幅: {((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100:+.2f}%")
print(f"平均波動: {((df['high'] - df['low']) / df['close'] * 100).mean():.2f}%")
print(f"上漲K線: {(df['close'] > df['open']).sum()} / {len(df)}")
print(f"平均成交量: {df['volume'].mean():.0f}")

# 執行市場結構分析 - 使用更敏感的參數
analyzer = OptimizedSmartMoneyConceptsAnalyzer()
# 使用更短的lookback期間，讓檢測更敏感
ms_results = analyzer.detect_market_structure(df, lookback=20)

print(f"\n=== CHoCH 市場結構分析結果 ===")

# 顯示CHoCH信號（主要關注看漲CHoCH）
choch_signals = [signal for signal in ms_results['choch_signals'] if 'BULLISH' in signal['type']]
print(f"\n1. 看漲結構改變 (CHoCH) - 檢測到 {len(choch_signals)} 個:")
print("=" * 80)
if choch_signals:
    for i, choch in enumerate(choch_signals, 1):
        print(f"{i}. {choch['type']}")
        print(f"   轉換價格: ${choch['price']:.2f}")
        print(f"   信號強度: {choch['strength']}/100")
        print(f"   描述: {choch['description']}")
        
        # 解析描述中的前低阻力價格
        if '前低阻力' in choch['description']:
            import re
            match = re.search(r'(\d+\.?\d*)', choch['description'])
            if match:
                prev_resistance = float(match.group(1))
                print(f"   前低阻力位: ${prev_resistance:.2f}")
        
        print("   " + "-" * 70)
        
        # 分析CHoCH品質
        if choch['strength'] >= 80:
            print(f"   💎 頂級CHoCH: 非常明確的趨勢轉換")
            print(f"   🚀 交易建議: 強烈建議進場，趨勢轉換確認")
        elif choch['strength'] >= 60:
            print(f"   🥇 優質CHoCH: 良好的趨勢轉換信號")
            print(f"   ✅ 交易建議: 可以進場，趨勢轉換機率高")
        elif choch['strength'] >= 40:
            print(f"   🥈 中等CHoCH: 一般的趨勢轉換")
            print(f"   ⚠️ 交易建議: 謹慎進場，等待更多確認")
        else:
            print(f"   🥉 弱勢CHoCH: 較弱的趨勢轉換")
            print(f"   ❌ 交易建議: 避免進場，信號不夠強")
        print()
else:
    print("   未檢測到看漲CHoCH信號")

# 當前趨勢
current_trend = ms_results.get('trend')
print(f"\n2. 當前市場趨勢:")
print("=" * 80)
if current_trend == 1:
    print("   🟢 看漲趨勢 - CHoCH已確認趨勢轉換")
    print("   ✅ 非常適合多頭交易策略")
elif current_trend == -1:
    print("   🔴 看跌趨勢 - 趨勢尚未轉換")
    print("   ❌ 不適合多頭交易")
else:
    print("   ⚪ 中性趨勢 - 趨勢轉換進行中")
    print("   ⏳ 密切關注CHoCH信號")

# 總結和交易建議
print(f"\n=== 趨勢轉換交易策略總結 ===")
if choch_signals:
    best_choch = max(choch_signals, key=lambda x: x['strength'])
    print(f"✅ 檢測到 {len(choch_signals)} 個看漲CHoCH信號")
    print(f"✅ 最強CHoCH信號強度: {best_choch['strength']}/100")
    print(f"✅ 趨勢轉換策略:")
    print(f"   1. CHoCH確認後，從看跌轉為看漲策略")
    print(f"   2. 在新趨勢的回調時積極進場")
    print(f"   3. 止損設在CHoCH前的結構點")
    print(f"   4. 目標價位放在新趨勢的延伸位置")
    
    if current_trend == 1:
        print(f"   5. 趨勢已確認轉換，保持多頭思維")
else:
    print("❌ 未檢測到看漲CHoCH信號")
    print("📊 市場可能仍處於原趨勢或整理階段")
    print("⏳ 繼續等待明確的趨勢轉換信號")

# 繪製CHoCH分析圖表
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[4, 1])

# 上圖：K線圖 + CHoCH標記
for i, (o, h, l, c) in enumerate(zip(df['open'], df['high'], df['low'], df['close'])):
    color = 'green' if c >= o else 'red'
    alpha = 0.8
    
    # 繪制影線
    ax1.plot([i, i], [l, h], color=color, linewidth=1, alpha=alpha)
    
    # 繪制實體
    body_height = abs(c - o)
    body_bottom = min(o, c)
    ax1.add_patch(patches.Rectangle((i-0.3, body_bottom), 0.6, body_height,
                                   facecolor=color, edgecolor=color, alpha=alpha))

# 標記重要的結構點
highs_mask = df['high'] == df['high'].rolling(window=8, center=True).max()
lows_mask = df['low'] == df['low'].rolling(window=8, center=True).min()

# 標記swing points
swing_high_indices = df[highs_mask].index
for idx in swing_high_indices:
    if idx > 4 and idx < len(df) - 4:
        ax1.scatter(idx, df['high'].iloc[idx], color='red', s=60, marker='^', 
                   label='Structure High' if idx == swing_high_indices[0] else "")

swing_low_indices = df[lows_mask].index
for idx in swing_low_indices:
    if idx > 4 and idx < len(df) - 4:
        ax1.scatter(idx, df['low'].iloc[idx], color='darkgreen', s=60, marker='v', 
                   label='Structure Low' if idx == swing_low_indices[0] else "")

# 標記CHoCH信號 - 用水平線標記最近的1個
if choch_signals:
    recent_choch = choch_signals[-1:]  # 只取最近1個
    for i, choch in enumerate(recent_choch):
        # 解析CHoCH描述以獲取被突破的前低阻力價格
        choch_level_price = choch['price']  # 默認使用信號價格
        
        if '前低阻力' in choch['description']:
            import re
            match = re.search(r'(\d+\.?\d*)', choch['description'])
            if match:
                choch_level_price = float(match.group(1))  # 使用前低阻力價格作為水平線
                print(f"CHoCH水平線將繪製在前低阻力: ${choch_level_price:.2f}")
        
        # 找到CHoCH水平線應該開始的位置（前低阻力形成的位置）
        choch_start_idx = 0
        tolerance = (df['high'].max() - df['low'].min()) * 0.002  # 0.2%容差
        
        # 找到前低阻力形成的K線
        for idx in range(len(df)):
            if abs(df['low'].iloc[idx] - choch_level_price) <= tolerance:
                choch_start_idx = idx
                break
        
        # 確保線條從前低阻力延伸到圖表右邊
        line_start = choch_start_idx
        line_end = len(df) - 1 + 5  # 延伸到圖表右邊
        
        # 畫CHoCH水平線 - 標記在前低阻力價位
        ax1.hlines(y=choch_level_price, xmin=line_start, xmax=line_end, 
                  colors='orange', linewidth=4, linestyle='--', alpha=1.0)
        
        # 在前低阻力位置標記CHoCH
        quality = "HIGH" if choch['strength'] >= 60 else "MED" if choch['strength'] >= 40 else "LOW"
        ax1.text(line_start + 5, choch_level_price - (df['high'].max() - df['low'].min()) * 0.01, 
                f'CHoCH\n{quality}\n${choch_level_price:.0f}', 
                ha='left', va='top', fontweight='bold', fontsize=9,
                color='orange', bbox=dict(boxstyle="round,pad=0.3", 
                facecolor='white', edgecolor='orange', alpha=0.9))

# 添加CHoCH解釋文字
ax1.text(0.75, 0.15, 'CHoCH (Change of Character):\nTrend direction changes\nFrom down to up trend', 
         transform=ax1.transAxes, fontsize=10, fontweight='bold', 
         verticalalignment='top', color='black',
         bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', 
                  edgecolor='orange', linewidth=1, alpha=0.8))

ax1.set_title("CHoCH (Change of Character) Analysis - Trend Reversal Focus", fontsize=16, fontweight='bold')
ax1.set_ylabel("Price ($)", fontsize=12)
ax1.legend(loc='upper left', fontsize=9)
ax1.grid(True, alpha=0.3)

# 添加當前趨勢狀態
trend_text = "🟢 BULLISH" if current_trend == 1 else "🔴 BEARISH" if current_trend == -1 else "⚪ NEUTRAL"
ax1.text(0.02, 0.98, f"Trend: {trend_text}", transform=ax1.transAxes, 
         fontsize=12, fontweight='bold', verticalalignment='top', color='black',
         bbox=dict(boxstyle="round,pad=0.5", facecolor='white', 
                  edgecolor='black', linewidth=1, alpha=0.9))

# 下圖：成交量分析
volume_bars = ax2.bar(range(len(df)), df['volume'], alpha=0.6)

# 突出CHoCH期間的爆量
volume_threshold_extreme = df['volume'].quantile(0.9)
volume_threshold_high = df['volume'].quantile(0.7)

for i, bar in enumerate(volume_bars):
    if df['volume'].iloc[i] > volume_threshold_extreme:
        bar.set_color('red')  # 爆量用紅色
        bar.set_alpha(1.0)
    elif df['volume'].iloc[i] > volume_threshold_high:
        bar.set_color('gold')  # 高量用金色
        bar.set_alpha(0.8)
    else:
        bar.set_color('lightblue')  # 正常量用淺藍
        bar.set_alpha(0.6)

ax2.set_title("Volume Analysis (Explosive Volume = CHoCH Confirmation)", fontsize=12)
ax2.set_ylabel("Volume", fontsize=10)
ax2.set_xlabel("Time (Candle Index)", fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('market_structure_CHoCH_analysis.png', dpi=150, bbox_inches='tight')
print("圖表已保存為 market_structure_CHoCH_analysis.png")

print(f"\n=== 測試完成 ===")
print("CHoCH市場結構分析已完成，圖表已保存")