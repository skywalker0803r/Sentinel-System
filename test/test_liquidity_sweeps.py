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

# ===== 獲取真實比特幣數據用於流動性掃蕩分析 =====
print("=== 流動性掃蕩分析 - 使用真實BTC數據 ===")
df = get_btc_data('BTC_USDT', '4h', 150)  # 獲取150根4小時K線，更適合掃蕩分析

print(f"\n真實BTC數據:")
print(df[['time', 'open', 'high', 'low', 'close', 'volume']].head(10))
print("...")
print(df[['time', 'open', 'high', 'low', 'close', 'volume']].tail(10))

print(f"\n=== 流動性掃蕩測試數據統計 ===")
print(f"數據期間: {df['time'].iloc[0]} 到 {df['time'].iloc[-1]}")
print(f"總K線數: {len(df)}")
print(f"價格範圍: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
print(f"總漲跌幅: {((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100:+.2f}%")
print(f"平均波動: {((df['high'] - df['low']) / df['close'] * 100).mean():.2f}%")
print(f"上漲K線: {(df['close'] > df['open']).sum()} / {len(df)}")

# 執行流動性掃蕩分析
analyzer = OptimizedSmartMoneyConceptsAnalyzer()
sweep_results = analyzer.detect_liquidity_sweeps(df)

print(f"\n=== 流動性掃蕩分析結果 ===")

# 只顯示買方流動性掃蕩（對多頭有利）
buy_side_sweeps = [sweep for sweep in sweep_results if sweep['type'] == 'BUY_SIDE_LIQUIDITY']

print(f"\n買方流動性掃蕩 - 檢測到 {len(buy_side_sweeps)} 個:")
print("=" * 90)

if buy_side_sweeps:
    for i, sweep in enumerate(buy_side_sweeps, 1):
        print(f"{i}. 買方流動性掃蕩")
        print(f"   時間: {sweep['time']}")
        print(f"   掃蕩價格: ${sweep['price']:.2f}")
        print(f"   被掃蕩支撐位: ${sweep['swept_level']:.2f}")
        print(f"   掃蕩距離: ${sweep['sweep_distance']:.2f}")
        print(f"   反彈幅度: ${sweep['max_recovery']:.2f}")
        print(f"   描述: {sweep['description']}")
        print("   " + "-" * 80)
        
        # 分析掃蕩品質
        sweep_percentage = (sweep['sweep_distance'] / sweep['swept_level']) * 100
        recovery_percentage = (sweep['max_recovery'] / sweep['price']) * 100
        
        print(f"   📊 掃蕩深度: {sweep_percentage:.3f}%")
        print(f"   📈 反彈幅度: {recovery_percentage:.3f}%")
        
        # 交易建議
        if recovery_percentage > 2.0 and sweep_percentage > 0.5:
            print(f"   💡 高品質掃蕩: 強烈的假突破信號")
            print(f"   🎯 交易機會: 在${sweep['swept_level']:.2f}附近等待反彈確認")
            print(f"   🛡️ 止損位置: ${sweep['price'] - sweep['sweep_distance'] * 0.5:.2f}")
            print(f"   🚀 目標價位: ${sweep['price'] + sweep['max_recovery'] * 1.2:.2f}")
        elif recovery_percentage > 1.0:
            print(f"   💡 中等掃蕩: 可考慮小倉位試探")
            print(f"   ⚠️ 謹慎進場: 等待更多確認信號")
        else:
            print(f"   ⚠️ 弱勢掃蕩: 反彈力度不足，謹慎對待")
        print()
else:
    print("   未檢測到買方流動性掃蕩")

# 分析掃蕩品質
if buy_side_sweeps:
    recoveries = [sweep['max_recovery'] for sweep in buy_side_sweeps]
    sweep_distances = [sweep['sweep_distance'] for sweep in buy_side_sweeps]
    
    print(f"\n=== 掃蕩品質統計 ===")
    print(f"平均掃蕩距離: ${np.mean(sweep_distances):.2f}")
    print(f"平均反彈幅度: ${np.mean(recoveries):.2f}")
    print(f"最大反彈幅度: ${np.max(recoveries):.2f}")
    
    # 品質分級
    high_quality = [s for s in buy_side_sweeps if (s['max_recovery']/s['price'])*100 > 2.0]
    medium_quality = [s for s in buy_side_sweeps if 1.0 <= (s['max_recovery']/s['price'])*100 <= 2.0]
    low_quality = [s for s in buy_side_sweeps if (s['max_recovery']/s['price'])*100 < 1.0]
    
    print(f"\n品質分級:")
    print(f"  🥇 高品質掃蕩 (反彈>2%): {len(high_quality)} 個")
    print(f"  🥈 中品質掃蕩 (反彈1-2%): {len(medium_quality)} 個")
    print(f"  🥉 低品質掃蕩 (反彈<1%): {len(low_quality)} 個")

# 多頭交易策略
print(f"\n=== 多頭交易策略 ===")
if buy_side_sweeps:
    high_quality_sweeps = [s for s in buy_side_sweeps if (s['max_recovery']/s['price'])*100 > 2.0]
    
    if high_quality_sweeps:
        best_sweep = max(high_quality_sweeps, key=lambda x: x['max_recovery'])
        print(f"✅ 發現 {len(high_quality_sweeps)} 個高品質買方流動性掃蕩")
        print(f"✅ 最佳交易機會: ${best_sweep['swept_level']:.2f} 支撐位")
        print(f"✅ 策略重點:")
        print(f"   1. 在支撐位附近等待價格回調")
        print(f"   2. 尋找反彈確認信號（如反轉K線模式）")
        print(f"   3. 在確認後進場做多")
        print(f"   4. 止損設在掃蕩低點下方")
        print(f"   5. 目標設在前高或阻力位")
    else:
        print(f"⚠️ 掃蕩品質一般，建議等待更好機會")
        print(f"📊 可關注後續價格行為確認反轉")
else:
    print("❌ 未檢測到買方流動性掃蕩")
    print("📈 目前價格行為未顯示明顯的流動性陷阱")

# 繪製流動性掃蕩圖表
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), height_ratios=[4, 1])

# 上圖：K線圖 + 流動性掃蕩
for i, (o, h, l, c) in enumerate(zip(df['open'], df['high'], df['low'], df['close'])):
    color = 'green' if c >= o else 'red'
    alpha = 0.8
    
    # 在掃蕩區域增強顯示
    if i in range(25, 40) or i in range(60, 75):
        alpha = 1.0
        linewidth = 1.5
    else:
        linewidth = 1
    
    # 繪制影線
    ax1.plot([i, i], [l, h], color=color, linewidth=linewidth, alpha=alpha)
    
    # 繪制實體
    body_height = abs(c - o)
    body_bottom = min(o, c)
    ax1.add_patch(patches.Rectangle((i-0.3, body_bottom), 0.6, body_height,
                                   facecolor=color, edgecolor=color, alpha=alpha))

# 標記買方流動性掃蕩 - 只顯示最新的1個
recent_sweeps = buy_side_sweeps[-1:] if buy_side_sweeps else []

for i, sweep in enumerate(recent_sweeps):
    # 找到對應的時間索引
    sweep_time = pd.to_datetime(sweep['time'])
    time_diff = abs(df['time'] - sweep_time)
    sweep_idx = time_diff.idxmin()
    
    # 繪制掃蕩箭頭和標記
    sweep_color = 'cyan'
    
    # 標記掃蕩點
    ax1.scatter(sweep_idx, sweep['price'], color=sweep_color, s=200, 
               marker='v', edgecolor='darkblue', linewidth=2, 
               label=f'Latest Liquidity Sweep' if i == 0 else "")
    
    # 標記被掃蕩的支撐位
    ax1.axhline(y=sweep['swept_level'], color=sweep_color, linestyle='--', 
               alpha=0.7, linewidth=2)
    
    # 添加文字說明
    recovery_pct = (sweep['max_recovery'] / sweep['price']) * 100
    quality = "HIGH" if recovery_pct > 2.0 else "MED" if recovery_pct > 1.0 else "LOW"
    
    ax1.annotate(f'BUY SWEEP\n${sweep["price"]:.0f}\nRecovery: {recovery_pct:.1f}%\nQuality: {quality}', 
                xy=(sweep_idx, sweep['price']),
                xytext=(sweep_idx - 8, sweep['price'] - (df['high'].max() - df['low'].min()) * 0.05),
                arrowprops=dict(arrowstyle='->', color=sweep_color, lw=2),
                color=sweep_color, fontweight='bold', ha='center', fontsize=9,
                bbox=dict(boxstyle="round,pad=0.4", facecolor='white', 
                         edgecolor=sweep_color, alpha=0.9))

# 標記一些關鍵時間區域（基於數據分析）
recent_volatility = df['volume'].rolling(10).std()
high_vol_periods = recent_volatility > recent_volatility.quantile(0.8)
if high_vol_periods.any():
    ax1.axvspan(0, len(df)//4, alpha=0.1, color='lightblue', label='Early Period')
    ax1.axvspan(len(df)//4, len(df)//2, alpha=0.1, color='lightgreen', label='Mid Period')
    ax1.axvspan(len(df)//2, 3*len(df)//4, alpha=0.1, color='gold', label='Late Period')
    ax1.axvspan(3*len(df)//4, len(df), alpha=0.1, color='lightcoral', label='Recent Period')

ax1.set_title("Liquidity Sweeps Analysis - Buy Side Focus", fontsize=14, fontweight='bold')
ax1.set_ylabel("Price ($)", fontsize=12)
ax1.legend(loc='upper left', fontsize=9)
ax1.grid(True, alpha=0.3)

# 下圖：成交量
volume_colors = ['green' if c >= o else 'red' for o, c in zip(df['open'], df['close'])]
bars = ax2.bar(range(len(df)), df['volume'], color=volume_colors, alpha=0.6)

# 突出顯示最近掃蕩時的成交量
for sweep in recent_sweeps:
    sweep_time = pd.to_datetime(sweep['time'])
    time_diff = abs(df['time'] - sweep_time)
    sweep_idx = time_diff.idxmin()
    
    # 高亮掃蕩時的成交量
    bars[sweep_idx].set_alpha(1.0)
    bars[sweep_idx].set_edgecolor('cyan')
    bars[sweep_idx].set_linewidth(3)

ax2.set_title("Volume (Sweep Events Highlighted)", fontsize=12)
ax2.set_ylabel("Volume", fontsize=10)
ax2.set_xlabel("Time (Candle Index)", fontsize=10)

plt.tight_layout()
plt.savefig('liquidity_sweeps_analysis.png', dpi=150, bbox_inches='tight')
print("圖表已保存為 liquidity_sweeps_analysis.png")
plt.show()

print(f"\n=== 測試完成 ===")
print("圖表已顯示並保存流動性掃蕩分析結果")