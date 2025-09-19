import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from get_real_data import get_btc_data
import warnings
import sys
import os
sys.path.append(os.path.abspath(".."))
from bot import calculate_price_skew_indicator,get_skew_level
# Suppress minor warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.rc('font', family='Microsoft JhengHei')

def plot_skew_analysis(df, skew_data, symbol, lookback_period=100):
    """
    Generates and saves a candlestick chart to visualize the price skew analysis.
    """
    fig, ax = plt.subplots(figsize=(15, 7))
    
    # This part remains the same
    plot_df = df.tail(lookback_period).reset_index(drop=True)

    # Plotting candlesticks
    for i, row in plot_df.iterrows():
        color = 'green' if row['close'] >= row['open'] else 'red'
        ax.plot([i, i], [row['low'], row['high']], color=color, linewidth=1)
        ax.add_patch(patches.Rectangle((i - 0.3, min(row['open'], row['close'])), 0.6, abs(row['close'] - row['open']),
                                       facecolor=color, edgecolor=color))

    # The rest of the function remains the same
    ax.legend()
    ax.set_title(f'Price Skew Analysis for {symbol} (Last {lookback_period} Candles)\nSkew Score: {skew_data["skew_score"]:.1f} - Level: {skew_data["skew_level"]}')
    ax.set_xlabel("Candle Index")
    ax.set_ylabel("Price ($)")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    filename = 'price_skew_analysis.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nChart saved as {filename}")
    plt.show()

if __name__ == "__main__":
    print("=== Price Skew Indicator Test ===")
    
    # --- Configuration ---
    # Using a more volatile altcoin to better demonstrate skewness
    # Fallback to BTC if the primary symbol fails
    SYMBOL_TO_TEST = 'GPT_USDT' 
    FALLBACK_SYMBOL = None
    DATA_POINTS = 300
    LOOKBACK = 100

    # --- Data Fetching ---
    df = get_btc_data(symbol=SYMBOL_TO_TEST, interval='1h', limit=DATA_POINTS)
    if df is None or len(df) < LOOKBACK:
        print(f"Could not fetch sufficient data for {SYMBOL_TO_TEST}. Falling back to {FALLBACK_SYMBOL}.")
        df = get_btc_data(symbol=FALLBACK_SYMBOL, interval='1h', limit=DATA_POINTS)

    if df is not None and len(df) >= LOOKBACK:
        print(f"\nSuccessfully loaded {len(df)} data points for {SYMBOL_TO_TEST if len(df) >= LOOKBACK else FALLBACK_SYMBOL}.")
        
        # --- Skew Calculation ---
        skew_analysis_result = calculate_price_skew_indicator(df, lookback_period=LOOKBACK)
        
        # --- Display Results ---
        if skew_analysis_result:
            print("\n--- Skew Analysis Results ---")
            print(f"Skew Score: {skew_analysis_result['skew_score']:.2f} / 100")
            print(f"Skew Level: {skew_analysis_result['skew_level']}")
            print(f"Is Skew Candidate: {'Yes' if skew_analysis_result['is_skew_candidate'] else 'No'}")
            print("-" * 30)
            print(f"High Price Skew Ratio (Mean/Median): {skew_analysis_result['high_skew_ratio']:.4f}")
            print(f"Extreme Upper Shadow Count (>50% of candle range): {skew_analysis_result['extreme_shadow_count']}")
            print(f"Extreme Upper Shadow Frequency: {skew_analysis_result['extreme_shadow_frequency']:.2%}")
            print(f"High Price Jump Frequency (>10%): {skew_analysis_result['jump_frequency']:.2%}")
            print("-----------------------------\n")

            # --- Visualization ---
            plot_skew_analysis(df, skew_analysis_result, SYMBOL_TO_TEST, lookback_period=LOOKBACK)
        else:
            print("Could not perform skew analysis due to insufficient data.")
    else:
        print("Failed to fetch any data. Cannot proceed with the test.")