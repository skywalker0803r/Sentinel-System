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

# ===== 獲取真實比特幣數據用於BOS分析 =====
print("=== BOS (Break of Structure) 分析 - 使用真實BTC數據 ===")
df = get_btc_data('BTC_USDT', '6h', 130)  # 獲取130根6小時K線

print(f"\n真實BTC數據:")
print(df[['time', 'open', 'high', 'low', 'close', 'volume']].head(10))
print("...")
print(df[['time', 'open', 'high', 'low', 'close', 'volume']].tail(10))

print(f"\n=== BOS (Break of Structure) 測試數據統計 ===")
print(f"數據期間: {df['time'].iloc[0]} 到 {df['time'].iloc[-1]}")
print(f"總K線數: {len(df)}")
print(f"價格範圍: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
print(f"總漲跌幅: {((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100:+.2f}%")
print(f"平均波動: {((df['high'] - df['low']) / df['close'] * 100).mean():.2f}%")
print(f"上漲K線: {(df['close'] > df['open']).sum()} / {len(df)}")
print(f"平均成交量: {df['volume'].mean():.0f}")

# 執行市場結構分析
analyzer = OptimizedSmartMoneyConceptsAnalyzer()
ms_results = analyzer.detect_market_structure(df)

print(f"\n=== BOS 市場結構分析結果 ===")

# 顯示BOS信號（主要關注看漲BOS）
bos_signals = [signal for signal in ms_results['bos_signals'] if 'BULLISH' in signal['type']]
print(f"\n1. 看漲突破結構 (BOS) - 檢測到 {len(bos_signals)} 個:")
print("=" * 80)
if bos_signals:
    for i, bos in enumerate(bos_signals, 1):
        print(f"{i}. {bos['type']}")
        print(f"   突破價格: ${bos['price']:.2f}")
        print(f"   信號強度: {bos['strength']}/100")
        print(f"   描述: {bos['description']}")
        
        # 解析描述中的前高價格
        if '前高' in bos['description']:
            import re
            match = re.search(r'(\d+\.?\d*)', bos['description'])
            if match:
                prev_high = float(match.group(1))
                print(f"   前高價位: ${prev_high:.2f}")
        
        print("   " + "-" * 70)
        
        # 分析BOS品質
        if bos['strength'] >= 80:
            print(f"   💎 頂級BOS: 非常強勢的結構突破")
            print(f"   🚀 交易建議: 積極進場，這是高品質的突破信號")
        elif bos['strength'] >= 60:
            print(f"   🥇 優質BOS: 良好的結構突破")
            print(f"   ✅ 交易建議: 可以進場，風險相對較低")
        elif bos['strength'] >= 40:
            print(f"   🥈 中等BOS: 一般的結構突破")
            print(f"   ⚠️ 交易建議: 謹慎進場，等待更多確認")
        else:
            print(f"   🥉 弱勢BOS: 較弱的結構突破")
            print(f"   ❌ 交易建議: 避免進場，等待更強信號")
        print()
else:
    print("   未檢測到看漲BOS信號")

# 顯示CHoCH信號
choch_signals = [signal for signal in ms_results['choch_signals'] if 'BULLISH' in signal['type']]
print(f"\n2. 看漲結構改變 (CHoCH) - 檢測到 {len(choch_signals)} 個:")
print("=" * 80)
if choch_signals:
    for i, choch in enumerate(choch_signals, 1):
        print(f"{i}. {choch['type']}")
        print(f"   轉換價格: ${choch['price']:.2f}")
        print(f"   信號強度: {choch['strength']}/100")
        print(f"   描述: {choch['description']}")
        print()

# 當前趨勢
current_trend = ms_results.get('trend')
print(f"\n3. 當前市場趨勢:")
print("=" * 80)
if current_trend == 1:
    print("   🟢 看漲趨勢 - 結構已確認向上")
    print("   ✅ 適合多頭交易策略")
elif current_trend == -1:
    print("   🔴 看跌趨勢 - 結構確認向下")
    print("   ❌ 不適合多頭交易")
else:
    print("   ⚪ 中性趨勢 - 結構不明確")
    print("   ⏳ 等待明確的結構確認")

# 總結和交易建議
print(f"\n=== 多頭交易策略總結 ===")
if bos_signals:
    best_bos = max(bos_signals, key=lambda x: x['strength'])
    print(f"✅ 檢測到 {len(bos_signals)} 個看漲BOS信號")
    print(f"✅ 最強BOS信號強度: {best_bos['strength']}/100")
    print(f"✅ 交易策略建議:")
    print(f"   1. BOS確認後積極尋找多頭機會")
    print(f"   2. 在回調至突破點附近時進場")
    print(f"   3. 止損設在BOS前的結構低點")
    print(f"   4. 目標價位設在下一個阻力區域")
    
    if current_trend == 1:
        print(f"   5. 當前趨勢配合，可保持多頭倉位")
else:
    print("❌ 未檢測到看漲BOS信號")
    print("📊 當前市場結構可能仍在整理階段")
    print("⏳ 建議等待明確的結構突破確認")

# 繪製BOS分析圖表
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[4, 1])

# 上圖：K線圖 + BOS標記
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

# 標記swing highs和lows
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

# 標記BOS信號 - 用水平線標記最近的1個
if bos_signals:
    recent_bos = bos_signals[-1:]  # 只取最近1個
    for i, bos in enumerate(recent_bos):
        # 解析BOS描述以獲取被突破的前高價格
        bos_level_price = bos['price']  # 默認使用信號價格
        
        if '前高' in bos['description']:
            import re
            match = re.search(r'(\d+\.?\d*)', bos['description'])
            if match:
                bos_level_price = float(match.group(1))  # 使用前高價格作為水平線
                print(f"BOS水平線將繪製在前高: ${bos_level_price:.2f}")
        
        # 找到BOS水平線應該開始的位置（前高形成的位置）
        bos_start_idx = 0
        tolerance = (df['high'].max() - df['low'].min()) * 0.002  # 0.2%容差
        
        # 找到前高形成的K線
        for idx in range(len(df)):
            if abs(df['high'].iloc[idx] - bos_level_price) <= tolerance:
                bos_start_idx = idx
                break
        
        # 確保線條從前高延伸到圖表右邊
        line_start = bos_start_idx
        line_end = len(df) - 1 + 5  # 延伸到圖表右邊
        
        # 畫BOS水平線 - 標記在前高價位
        ax1.hlines(y=bos_level_price, xmin=line_start, xmax=line_end, 
                  colors='lime', linewidth=4, linestyle='-', alpha=1.0)
        
        # 在前高位置標記BOS
        quality = "HIGH" if bos['strength'] >= 60 else "MED" if bos['strength'] >= 40 else "LOW"
        ax1.text(line_start + 5, bos_level_price + (df['high'].max() - df['low'].min()) * 0.01, 
                f'BOS\n{quality}\n${bos_level_price:.0f}', 
                ha='left', va='bottom', fontweight='bold', fontsize=9,
                color='lime', bbox=dict(boxstyle="round,pad=0.3", 
                facecolor='white', edgecolor='lime', alpha=0.9))

# 添加BOS解釋文字
ax1.text(0.75, 0.15, 'BOS (Break of Structure):\nPrice breaks previous high\nConfirms trend continuation', 
         transform=ax1.transAxes, fontsize=10, fontweight='bold', 
         verticalalignment='top', color='black',
         bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', 
                  edgecolor='green', linewidth=1, alpha=0.8))

ax1.set_title("BOS (Break of Structure) Analysis - Bullish Focus", fontsize=16, fontweight='bold')
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

# 用顏色標記不同階段的成交量
volume_threshold_high = df['volume'].quantile(0.8)
volume_threshold_med = df['volume'].quantile(0.6)

for i, bar in enumerate(volume_bars):
    if df['volume'].iloc[i] > volume_threshold_high:
        bar.set_color('red')  # 超高成交量用紅色
        bar.set_alpha(0.9)
    elif df['volume'].iloc[i] > volume_threshold_med:
        bar.set_color('gold')  # 高成交量用金色
        bar.set_alpha(0.8)
    else:
        bar.set_color('lightblue')  # 正常成交量用淺藍
        bar.set_alpha(0.6)

ax2.set_title("Volume Analysis (High Volume = Potential BOS)", fontsize=12)
ax2.set_ylabel("Volume", fontsize=10)
ax2.set_xlabel("Time (Candle Index)", fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('market_structure_BOS_analysis.png', dpi=150, bbox_inches='tight')
print("圖表已保存為 market_structure_BOS_analysis.png")

print(f"\n=== 測試完成 ===")
print("BOS市場結構分析已完成，圖表已保存")