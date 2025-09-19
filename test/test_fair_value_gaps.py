import sys
import os
import numpy as np
# 把上一層資料夾加入 Python 搜尋路徑
sys.path.append(os.path.abspath(".."))
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from smc_indicators_optimized import OptimizedSmartMoneyConceptsAnalyzer
from get_real_data import get_btc_data
import warnings 
warnings.filterwarnings('ignore')

# ===== 獲取真實比特幣數據 =====
print("=== Fair Value Gaps (FVG) 分析 - 使用真實BTC數據 ===")
# 嘗試不同時間框架來找到FVG
print("正在嘗試不同時間框架來檢測FVG...")

# 先嘗試15分鐘K線（更容易出現FVG）
df = get_btc_data('BTC_USDT', '15m', 200)
analyzer = OptimizedSmartMoneyConceptsAnalyzer()
fvg_list = analyzer.detect_fair_value_gaps(df)

if len(fvg_list) == 0:
    print("15分鐘數據未檢測到FVG，嘗試5分鐘數據...")
    df = get_btc_data('BTC_USDT', '5m', 800)
    fvg_list = analyzer.detect_fair_value_gaps(df)

if len(fvg_list) == 0:
    print("5分鐘數據未檢測到FVG，嘗試1分鐘數據...")
    df = get_btc_data('BTC_USDT', '1m', 1000)
    fvg_list = analyzer.detect_fair_value_gaps(df)

print(f"使用時間框架: {df['time'].iloc[1] - df['time'].iloc[0]} 間隔")
print(f"最終檢測結果: {len(fvg_list)} 個FVG")

print(f"\n真實BTC數據:")
print(df[['time', 'open', 'high', 'low', 'close', 'volume']].head(10))
print("...")
print(df[['time', 'open', 'high', 'low', 'close', 'volume']].tail(10))

print(f"\n數據統計:")
print(f"數據期間: {df['time'].iloc[0]} 到 {df['time'].iloc[-1]}")
print(f"價格範圍: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
print(f"平均波動: {((df['high'] - df['low']) / df['close'] * 100).mean():.2f}%")
print(f"上漲K線: {(df['close'] > df['open']).sum()} / {len(df)}")
print(f"總漲跌幅: {((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100:+.2f}%")

# analyzer 和 fvg_list 已在上面設定

print(f"\n檢測到 {len(fvg_list)} 個看漲Fair Value Gaps:")
print("=" * 80)
for i, fvg in enumerate(fvg_list, 1):
    print(f"{i}. {fvg['type']}")
    print(f"   時間: {fvg['time']}")
    print(f"   價格區間: ${fvg['low']:.2f} - ${fvg['high']:.2f}")
    print(f"   缺口大小: ${fvg['size']:.2f} ({fvg['size']/fvg['low']*100:.3f}%)")
    print(f"   狀態: {'✅ 未填補' if not fvg['filled'] else '❌ 已填補'}")
    print("   " + "-" * 50)

if len(fvg_list) == 0:
    print("   未檢測到任何看漲FVG")

print(f"\n多頭交易建議:")
print("✅ 可在FVG區域尋找支撐進場機會")
print("✅ 設置止損在FVG下方")
print("✅ 目標價位在下一個阻力位")

# ===== 畫蠟燭圖 =====
fig, ax = plt.subplots(figsize=(14, 6))

for i, (o, h, l, c) in enumerate(zip(df['open'], df['high'], df['low'], df['close'])):
    color = 'green' if c >= o else 'red'
    ax.plot([i, i], [l, h], color=color, linewidth=1)
    ax.add_patch(patches.Rectangle((i-0.3, min(o, c)), 0.6, abs(c-o),
                                   facecolor=color, edgecolor=color))

# 標記 FVG - 只顯示最新的1個
recent_fvgs = fvg_list[:] if fvg_list else []

for i, fvg in enumerate(recent_fvgs):
    # 用 iloc 找對應 index
    fvg_time_idx = df[df['time'] == fvg['time']].index
    if len(fvg_time_idx) > 0:
        idx = fvg_time_idx[0]
        
        # 只顯示看漲FVG（綠色）
        fvg_color = 'lime'
        
        ax.add_patch(patches.Rectangle(
            (idx-1, fvg['low']), 5,  # 寬度覆蓋 5 根 K 線
            fvg['high'] - fvg['low'],
            facecolor=fvg_color, alpha=0.3, edgecolor=fvg_color, linewidth=2
        ))
        
        # 添加文字標記
        ax.text(idx, fvg['high'] + (df['high'].max() - df['low'].min()) * 0.01, 
                f'Latest FVG\n${fvg["size"]:.0f}', 
                color=fvg_color, fontweight='bold', fontsize=9,
                ha='center', va='bottom',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

ax.set_title("Fair Value Gaps (FVG) - Bullish Only")
ax.set_xlabel("Candle Index")
ax.set_ylabel("Price ($)")
plt.savefig('fair_value_gaps_analysis.png', dpi=150, bbox_inches='tight')
print("圖表已保存為 fair_value_gaps_analysis.png")
plt.show()

print(fvg_list)