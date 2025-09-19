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
import re
warnings.filterwarnings('ignore')

# --- 1. 獲取大量歷史數據 ---
print("=== 開始尋找最優的下降趨勢反轉場景 ===")
df = get_btc_data('TCOM_USDT', '1h', 5000) # 獲取5000根K線 (約3.4年數據)
if df is None or len(df) < 200:
    print("數據不足，無法進行分析。")
    sys.exit()

# --- 2. 搜集所有相關歷史信號 ---
analyzer = OptimizedSmartMoneyConceptsAnalyzer()
all_bearish_bos = []
all_bullish_choch = []
lookback = 20

print("\n=== 正在遍歷海量歷史數據，搜集所有潛在信號... ===")
for i in range(100, len(df), 10):
    df_slice = df.iloc[:i].copy()
    ms_results = analyzer.detect_market_structure(df_slice, lookback=lookback)
    
    if ms_results.get('bos_signals'):
        for bos in ms_results['bos_signals']:
            if bos['type'] == 'BOS_BEARISH' and not any(np.isclose(s['price'], bos['price']) for s in all_bearish_bos):
                bos['detected_at_index'] = i
                all_bearish_bos.append(bos)
    
    if ms_results.get('choch_signals'):
        for choch in ms_results['choch_signals']:
            if choch['type'] == 'CHOCH_BULLISH' and not any(np.isclose(s['price'], choch['price']) for s in all_bullish_choch):
                choch['detected_at_index'] = i
                all_bullish_choch.append(choch)

print(f"搜集完成: 共找到 {len(all_bearish_bos)} 個看跌BOS信號 和 {len(all_bullish_choch)} 個看漲CHoCH信號。")

# --- 3. 構建並評分所有可能的場景 ---
scenarios = []
print("\n=== 正在構建並評分所有 '2看跌BOS -> 1看漲CHoCH' 的場景... ===")
if all_bullish_choch and len(all_bearish_bos) >= 2:
    for choch in all_bullish_choch:
        preceding_bos = [bos for bos in all_bearish_bos if bos['detected_at_index'] < choch['detected_at_index']]
        if len(preceding_bos) >= 2:
            bos2 = sorted(preceding_bos, key=lambda x: x['detected_at_index'])[-1]
            bos1 = sorted(preceding_bos, key=lambda x: x['detected_at_index'])[-2]

            score = 0
            try:
                bos1_level = float(re.search(r'(\d+\.?\d*)', bos1['description']).group(1))
                bos2_level = float(re.search(r'(\d+\.?\d*)', bos2['description']).group(1))
                lh_level = float(re.search(r'(\d+\.?\d*)', choch['description']).group(1))
                
                if bos1_level > 0 and lh_level > 0:
                    trend_score = (bos1_level - bos2_level) / bos1_level
                    score += trend_score * 50

                    reversal_score = (choch['price'] - lh_level) / lh_level
                    score += reversal_score * 50

                    scenarios.append({
                        'signals': [bos1, bos2, choch],
                        'score': score
                    })
            except (AttributeError, IndexError):
                continue

# --- 4. 選出最佳場景並準備繪圖 ---
signals_to_plot = []
if scenarios:
    best_scenario = sorted(scenarios, key=lambda x: x['score'], reverse=True)[0]
    signals_to_plot = best_scenario['signals']
    print(f"🎉 成功找到最佳場景！得分: {best_scenario['score']:.2f}。將繪製3個信號。 সন")
else:
    print("❌ 未能從歷史數據中構建出任何符合條件的場景。")

# --- 5. 繪製最佳場景 ---
if signals_to_plot:
    print("\n=== 計算繪圖的縮放範圍... ===")
    all_indices = []
    for signal in signals_to_plot:
        all_indices.append(signal['detected_at_index'])
        level_price = float(re.search(r'(\d+\.?\d*)', signal['description']).group(1))
        
        if 'BOS' in signal['type']:
            structure_idx = (df['low'] - level_price).abs().idxmin()
        else:
            structure_idx = (df['high'] - level_price).abs().idxmin()
        all_indices.append(structure_idx)
    
    min_idx = min(all_indices)
    max_idx = max(all_indices)
    
    buffer = 30
    plot_start_idx = max(0, min_idx - buffer)
    plot_end_idx = min(len(df), max_idx + buffer)
    
    plot_df = df.iloc[plot_start_idx:plot_end_idx].copy()
    print(f"🔎 成功！將繪圖範圍縮放至索引 {plot_start_idx} 到 {plot_end_idx}。")

    print("\n📊 正在生成最終圖表...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), height_ratios=[4, 1])

    for k, (o, h, l, c) in enumerate(zip(plot_df['open'], plot_df['high'], plot_df['low'], plot_df['close'])):
        color = 'green' if c >= o else 'red'
        ax1.plot([k, k], [l, h], color=color, linewidth=1, alpha=0.8)
        ax1.add_patch(patches.Rectangle((k-0.3, min(o,c)), 0.6, abs(c-o), facecolor=color, edgecolor=color, alpha=0.8))

    labeled_points = set()
    for signal in signals_to_plot:
        level_price = float(re.search(r'(\d+\.?\d*)', signal['description']).group(1))
        breakout_idx_orig = signal['detected_at_index']

        if signal['type'] == 'BOS_BEARISH':
            structure_idx_orig = (df['low'] - level_price).abs().idxmin()
            start_pos_local = structure_idx_orig - plot_start_idx
            end_pos_local = breakout_idx_orig - plot_start_idx
            
            if start_pos_local >= 0 and end_pos_local < len(plot_df):
                ax1.hlines(y=level_price, xmin=start_pos_local, xmax=end_pos_local, colors='red', linewidth=2, linestyle='-')
                ax1.text(start_pos_local + (end_pos_local - start_pos_local) / 2, level_price - (plot_df['high'].max() - plot_df['low'].min())*0.02, 'BOS', ha='center', va='top', color='red', fontweight='bold', fontsize=12)
                
                if structure_idx_orig not in labeled_points:
                    ax1.text(start_pos_local, level_price, 'LL', ha='center', va='top', color='black', fontweight='bold', fontsize=12)
                    labeled_points.add(structure_idx_orig)

        elif signal['type'] == 'CHOCH_BULLISH':
            structure_idx_orig = (df['high'] - level_price).abs().idxmin()
            start_pos_local = structure_idx_orig - plot_start_idx
            end_pos_local = breakout_idx_orig - plot_start_idx

            if start_pos_local >= 0 and end_pos_local < len(plot_df):
                ax1.hlines(y=level_price, xmin=start_pos_local, xmax=end_pos_local, colors='orange', linewidth=2, linestyle='--')
                ax1.text(start_pos_local + (end_pos_local - start_pos_local) / 2, level_price + (plot_df['high'].max() - plot_df['low'].min())*0.02, 'CHOCH', ha='center', va='bottom', color='orange', fontweight='bold', fontsize=12)

                if structure_idx_orig not in labeled_points:
                    ax1.text(start_pos_local, level_price, 'LH', ha='center', va='bottom', color='black', fontweight='bold', fontsize=12)
                    labeled_points.add(structure_idx_orig)

    ax1.set_title("Market Structure: Best Downtrend Reversal Scenario", fontsize=16, fontweight='bold')
    ax1.set_ylabel("Price ($)")
    ax1.grid(True, alpha=0.3)
    
    ax2.bar(range(len(plot_df)), plot_df['volume'], color='grey', alpha=0.5)
    ax2.set_ylabel("Volume")
    ax2.set_xlabel("Time (Candle Index)")

    plt.tight_layout()
    plt.savefig('market_structure_scenario.png', dpi=150, bbox_inches='tight')
    print("圖表已保存為 market_structure_scenario.png")
else:
    print("\n🤷‍♀️ 在整個數據集中沒有找到符合條件的信號。")

print(f"\n=== 測試完成 ===")
