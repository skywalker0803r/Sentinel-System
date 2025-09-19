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

# ===== 獲取真實比特幣數據用於Order Blocks分析 =====
print("=== Order Blocks 分析 - 使用真實BTC數據 ===")
df = get_btc_data('BTC_USDT', '2h', 180)  # 獲取180根2小時K線

print(f"\n真實BTC數據:")
print(df[['time', 'open', 'high', 'low', 'close', 'volume']].head(10))
print("...")
print(df[['time', 'open', 'high', 'low', 'close', 'volume']].tail(10))

print(f"\n=== Order Blocks 測試數據統計 ===")
print(f"數據期間: {df['time'].iloc[0]} 到 {df['time'].iloc[-1]}")
print(f"總K線數: {len(df)}")
print(f"價格範圍: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
print(f"總漲跌幅: {((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100:+.2f}%")
print(f"平均波動: {((df['high'] - df['low']) / df['close'] * 100).mean():.2f}%")
print(f"上漲K線: {(df['close'] > df['open']).sum()} / {len(df)}")
print(f"平均成交量: {df['volume'].mean():.0f}")
print(f"成交量峰值: {df['volume'].max():.0f}")

# 執行Order Blocks分析
analyzer = OptimizedSmartMoneyConceptsAnalyzer()
ob_results = analyzer.detect_order_blocks(df)

print(f"\n=== Order Blocks 分析結果 ===")

# 只顯示看漲Order Blocks
bullish_obs = [ob for ob in ob_results if ob['type'] == 'BULLISH_OB']

print(f"\n看漲Order Blocks - 檢測到 {len(bullish_obs)} 個:")
print("=" * 80)

if bullish_obs:
    for i, ob in enumerate(bullish_obs, 1):
        ob_size = ob['high'] - ob['low']
        ob_percentage = (ob_size / ob['low']) * 100
        
        print(f"{i}. 看漲Order Block")
        print(f"   時間: {ob['time']}")
        print(f"   價格區間: ${ob['low']:.2f} - ${ob['high']:.2f}")
        print(f"   區塊大小: ${ob_size:.2f} ({ob_percentage:.3f}%)")
        print(f"   強度評分: {ob['strength']}/100")
        print(f"   狀態: {'🟢 活躍' if ob['active'] else '🔴 非活躍'}")
        print(f"   描述: {ob['description']}")
        print("   " + "-" * 70)
        
        # 添加交易建議
        if ob['active'] and ob['strength'] > 60:
            print(f"   💡 交易建議: 高品質OB，適合在${ob['low']:.2f}附近尋找多頭進場")
            print(f"   🎯 目標價位: ${ob['high'] + ob_size * 0.5:.2f}+")
            print(f"   🛡️ 止損建議: ${ob['low'] - ob_size * 0.3:.2f}")
        elif ob['active']:
            print(f"   💡 交易建議: 中等品質OB，謹慎進場")
        else:
            print(f"   ⚠️ 注意: OB已失效，避免在此區域建倉")
        print()
else:
    print("   未檢測到任何看漲Order Blocks")

# 分析Order Blocks品質分布
if bullish_obs:
    strengths = [ob['strength'] for ob in bullish_obs]
    active_count = sum(1 for ob in bullish_obs if ob['active'])
    
    print(f"\n=== Order Blocks 品質分析 ===")
    print(f"平均強度: {np.mean(strengths):.1f}/100")
    print(f"最高強度: {np.max(strengths)}/100")
    print(f"活躍OB數量: {active_count}/{len(bullish_obs)}")
    
    # 品質分級
    high_quality = [ob for ob in bullish_obs if ob['strength'] >= 70]
    medium_quality = [ob for ob in bullish_obs if 50 <= ob['strength'] < 70]
    low_quality = [ob for ob in bullish_obs if ob['strength'] < 50]
    
    print(f"\n品質分級:")
    print(f"  🥇 高品質 (≥70): {len(high_quality)} 個")
    print(f"  🥈 中品質 (50-69): {len(medium_quality)} 個") 
    print(f"  🥉 低品質 (<50): {len(low_quality)} 個")

# 多頭交易總結
print(f"\n=== 多頭交易策略建議 ===")
if bullish_obs:
    active_obs = [ob for ob in bullish_obs if ob['active']]
    if active_obs:
        best_ob = max(active_obs, key=lambda x: x['strength'])
        print(f"✅ 發現 {len(active_obs)} 個活躍的看漲Order Block")
        print(f"✅ 最佳進場區域: ${best_ob['low']:.2f} - ${best_ob['high']:.2f}")
        print(f"✅ 建議策略: 在OB區域等待確認信號後進場")
        print(f"✅ 風險管理: 止損設在OB下方，目標設在上方阻力位")
    else:
        print("⚠️ 所有Order Block均已失效")
        print("⏳ 等待新的反轉結構形成")
else:
    print("❌ 未檢測到看漲Order Block")
    print("📊 建議等待明確的需求區域形成")

# 繪製Order Blocks圖表
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), height_ratios=[4, 1])

# 上圖：K線圖 + Order Blocks
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

# 標記Order Blocks - 只顯示最新的1個活躍OB
active_obs = [ob for ob in bullish_obs if ob['active']]
recent_obs = active_obs[-1:] if active_obs else []  # 只取最新1個

colors = ['lime']
for i, ob in enumerate(recent_obs):
    # 找到對應的時間索引
    ob_time = pd.to_datetime(ob['time'])
    time_diff = abs(df['time'] - ob_time)
    ob_idx = time_diff.idxmin()
    
    color = colors[i % len(colors)]
    alpha = 0.7  # 統一使用較高透明度
    
    # Order Block - 找到區域內最低K線，從下影線到實體
    # 獲取OB檢測的價格範圍
    ob_range_low = ob['low']
    ob_range_high = ob['high']
    
    # 在OB價格範圍內找到最低的K線
    # 找到在OB時間點附近的K線範圍
    search_range = 10  # 前後搜索10根K線
    start_idx = max(0, ob_idx - search_range)
    end_idx = min(len(df), ob_idx + search_range)
    
    # 在這個範圍內找到最低點的K線
    lowest_idx = start_idx
    lowest_low = df['low'].iloc[start_idx]
    
    for idx in range(start_idx, end_idx):
        if df['low'].iloc[idx] < lowest_low:
            lowest_low = df['low'].iloc[idx]
            lowest_idx = idx
    
    # 獲取最低K線的資料
    lowest_candle_low = df['low'].iloc[lowest_idx]      # 下影線低點
    lowest_candle_open = df['open'].iloc[lowest_idx]
    lowest_candle_close = df['close'].iloc[lowest_idx]
    
    # OB區域：從下影線低點到實體頂部
    ob_bottom = lowest_candle_low
    ob_top = max(lowest_candle_open, lowest_candle_close)  # 實體頂部
    
    print(f"DEBUG: 找到最低K線在索引 {lowest_idx}")
    print(f"DEBUG: OB區域 ${ob_bottom:.2f} (下影線) - ${ob_top:.2f} (實體頂)")
    
    # 繪制OB長方形區域 - 綠色透明
    rect_width = 15  # 覆蓋15根K線，形成明顯的區域
    rect = patches.Rectangle((ob_idx - rect_width//2, ob_bottom), 
                           rect_width, ob_top - ob_bottom,
                           facecolor='lime', alpha=0.2, 
                           edgecolor='lime', linewidth=2)
    ax1.add_patch(rect)
    
    # 添加標籤（放在OB區域上方）
    label_text = f"Latest OB\nStr:{ob['strength']}"
        
    ax1.text(ob_idx, ob_top + (df['high'].max() - df['low'].min()) * 0.01,
             label_text, ha='center', va='bottom', 
             color='lime', fontweight='bold', fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                      edgecolor='lime', alpha=0.9))

ax1.set_title("Order Blocks Analysis - Bullish Focus", fontsize=14, fontweight='bold')
ax1.set_ylabel("Price ($)", fontsize=12)
ax1.grid(True, alpha=0.3)

# 添加圖例
legend_elements = [
    patches.Patch(color='lime', alpha=0.6, label='Active Bullish OB'),
    patches.Patch(color='lime', alpha=0.3, label='Inactive Bullish OB')
]
ax1.legend(handles=legend_elements, loc='upper left')

# 下圖：成交量
volume_colors = ['green' if c >= o else 'red' for o, c in zip(df['open'], df['close'])]
bars = ax2.bar(range(len(df)), df['volume'], color=volume_colors, alpha=0.6)

# 突出顯示最近OB位置的成交量
for ob in recent_obs:
    ob_time = pd.to_datetime(ob['time'])
    time_diff = abs(df['time'] - ob_time)
    ob_idx = time_diff.idxmin()
    bars[ob_idx].set_alpha(1.0)
    bars[ob_idx].set_edgecolor('yellow')
    bars[ob_idx].set_linewidth(2)

ax2.set_title("Volume (OB Locations Highlighted)", fontsize=12)
ax2.set_ylabel("Volume", fontsize=10)
ax2.set_xlabel("Time (Candle Index)", fontsize=10)

plt.tight_layout()
plt.savefig('order_blocks_analysis.png', dpi=150, bbox_inches='tight')
print("圖表已保存為 order_blocks_analysis.png")
plt.show()

print(f"\n=== 測試完成 ===")
print("圖表已顯示並保存Order Blocks分析結果")