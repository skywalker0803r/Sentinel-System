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
df = get_btc_data('MAV_USDT', '1h', 180)  # 獲取180根2小時K線

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
order_blocks = bullish_obs
# 繪製Order Blocks圖表
if order_blocks: # 只有當檢測到Order Block時才繪圖
    latest_ob = order_blocks[-1] # 獲取最新偵測到的Order Block
    
    # 確定最新OB的K線索引
    ob_time_idx = df[df['time'] == latest_ob['time']].index
    if len(ob_time_idx) > 0:
        ob_idx = ob_time_idx[0]
        
        # 定義繪圖範圍：OB前後各50根K線
        buffer = 50 
        plot_start_idx = max(0, ob_idx - buffer)
        plot_end_idx = min(len(df), ob_idx + buffer + 1) # +1 確保包含結束K線
        
        plot_df = df.iloc[plot_start_idx:plot_end_idx].copy()
        
        fig, ax = plt.subplots(figsize=(14, 6)) # 使用單一子圖
        
        # 繪製縮放後的K線圖
        for i, (o, h, l, c) in enumerate(zip(plot_df['open'], plot_df['high'], plot_df['low'], plot_df['close'])):
            color = 'green' if c >= o else 'red'
            ax.plot([i, i], [l, h], color=color, linewidth=1)
            ax.add_patch(patches.Rectangle((i-0.3, min(o, c)), 0.6, abs(c-o), facecolor=color, edgecolor=color))

        # 繪製Order Blocks (只繪製在縮放範圍內的OB)
                # 繪製最新的Order Block
        ob = latest_ob # 直接使用latest_ob
        
        ob_orig_idx = df[df['time'] == ob['time']].index[0]
        # 確保最新的OB在縮放範圍內 (理論上應該是，因為我們就是圍繞它縮放的)
        if plot_start_idx <= ob_orig_idx < plot_end_idx:
            ob_local_idx = ob_orig_idx - plot_start_idx
            
            ob_color = 'blue' if 'BULLISH' in ob['type'] else 'red'
            
            # 計算寬度以延伸到右邊緣
            rect_width = len(plot_df) - ob_local_idx # 從OB位置延伸到plot_df的末尾
            
            ax.add_patch(patches.Rectangle(
                (ob_local_idx - 0.5, ob['low']), rect_width, # 使用計算出的寬度
                ob['high'] - ob['low'],
                facecolor=ob_color, alpha=0.3, edgecolor=ob_color, linewidth=1
            ))
            
            # 添加文字標記
            ax.text(ob_local_idx, ob['high'] + (plot_df['high'].max() - plot_df['low'].min()) * 0.01,
                    'OB', 
                    color=ob_color, fontweight='bold', fontsize=9,
                    ha='center', va='bottom')
        
        ax.set_title(f"Order Blocks (OB) - Latest OB at {latest_ob['time']}")
        ax.set_xlabel("Candle Index (Zoomed)")
        ax.set_ylabel("Price ($)")
        plt.savefig('order_blocks_analysis.png', dpi=150, bbox_inches='tight')
        print("圖表已保存為 order_blocks_analysis.png")
        plt.show()
    else:
        print("未找到最新Order Block的對應K線索引。")
else:
    print("未檢測到任何Order Block，無法繪圖。")

print(order_blocks)