# Sentinel-System - Vegas 通道 + Smart Money Concepts 加密貨幣交易訊號機器人

## 🎯 專案簡介

Sentinel-System 是一個高度集成的 Discord 交易機器人，結合了 **Vegas 通道** 和 **Smart Money Concepts (SMC)** 兩大技術分析體系，為加密貨幣交易者提供專業級的多重確認交易訊號。透過 Gate.io API 獲取實時市場數據，運用 APY 智能篩選和綜合評分系統，自動識別高潛力的交易機會。

## ✨ 核心功能特色

### 🔥 雙重技術分析系統
- **Vegas 通道策略**：基於 EMA12/144/169 的經典突破反彈系統
- **Smart Money Concepts**：機構級價格行為分析
- **多重確認機制**：Vegas + SMC 雙重驗證，提高訊號可靠性

### 🧠 Smart Money Concepts 完整功能
- **🏗️ 市場結構 (Market Structure)**
  - BOS (Break of Structure) - 結構突破檢測
  - CHoCH (Change of Character) - 趨勢轉變檢測
- **📦 Order Blocks (機構訂單區塊)**
  - 智能識別機構大單區域
  - 活躍/失效狀態實時監控
- **💎 Fair Value Gaps (公允價值缺口)**
  - 價格缺口自動檢測
  - 填補狀態追踪
- **⚖️ Equal Highs/Lows (等高點/等低點)**
  - 關鍵支撐阻力位識別
- **⚡ 流動性掃蕩 (Liquidity Sweeps)**
  - 假突破/假跌破檢測
  - 市場操縱模式識別
- **🎯 Premium/Discount 價格區間**
  - 🔴 **高價區 (70-100%)**: 賣出/止盈區域
  - 🟢 **低價區 (0-30%)**: 買入/加倉區域
  - 🟡 **平衡區 (30-70%)**: 觀望區域

### 📊 智能篩選與評分系統
- **APY 智能篩選**：自動篩選前 20% 高 APY 幣種（通常為低點潛力標的）
- **綜合評分 (0-100分)**：
  - Vegas 通道：25分（突破）/ 15分（反彈）
  - SMC 結構：15分（BOS）/ 20分（CHoCH）
  - Order Blocks：最高 15分
  - Fair Value Gaps：最高 10分
  - 流動性掃蕩：最高 10分
  - 高年利率：最高 10分

### 🏆 分層訊號展示
- **TOP 10 做多推薦**：專注高潛力多頭機會
- **動態分割顯示**：避免 Discord 字符限制
- **詳細價格區間**：每個幣種顯示具體高價區和低價區
- **評分明細**：透明化評分邏輯，便於理解

### 🚀 高性能技術架構
- **非同步處理**：高效並發 API 調用
- **智能速率控制**：遵守 Gate.io API 限制
- **記憶體優化**：逐一處理避免記憶體溢出
- **錯誤容錯**：完整異常處理機制

## 📁 專案結構

```
Sentinel-System/
├── bot.py                    # 主要機器人程式
├── smc_indicators.py         # SMC 核心分析器
├── smc_config.py            # SMC 配置參數
├── requirements.txt         # Python 依賴包
├── .env                     # 環境變數配置
├── README.md               # 專案說明文件
├── SMC_Integration_Summary.md        # SMC 整合報告
├── SMC_Price_Zones_Enhancement.md   # 價格區間功能說明
└── Discord_Field_Limit_Fix.md       # Discord 限制修復說明
```

## 🛠️ 安裝與設定

### 1. 環境準備
確保您的系統已安裝 **Python 3.8+**

### 2. 安裝依賴
```bash
git clone https://github.com/your-repo/Sentinel-System
cd Sentinel-System
pip install -r requirements.txt
```

### 3. 環境變數設定
創建 `.env` 文件並配置：
```env
DISCORD_TOKEN=你的Discord機器人TOKEN
CHANNEL_ID=你的Discord頻道ID
```

### 4. Discord 機器人設定
1. 前往 [Discord Developer Portal](https://discord.com/developers/applications)
2. 創建新應用程式 → Bot → 複製 TOKEN
3. 啟用 `Message Content Intent`
4. 在伺服器中啟用開發者模式，複製頻道 ID

### 5. 運行機器人
```bash
python bot.py
```

## 📋 使用說明

### 自動運行模式
機器人啟動後會自動執行：
1. **APY 篩選**：掃描所有 USDT 交易對，篩選前 20% 高 APY 幣種
2. **技術分析**：對篩選後的幣種進行 Vegas + SMC 雙重分析
3. **訊號評分**：計算每個訊號的綜合評分
4. **結果推送**：將 TOP 10 做多推薦發送到 Discord

### Discord 顯示範例
```
🚀 TOP 10 做多訊號分析
專注做多機會 - Vegas 通道 + Smart Money Concepts

📊 做多訊號統計
做多訊號數: 10
Vegas+SMC: 8
純SMC訊號: 2
平均評分: 65.2/100

🏆 TOP 做多推薦 (1-2)
🥇 1. SPA_USDT 🚀 75分
💰 $0.010768 | 📊 向上突破 | 🏦 10.58%
🎯 趨勢轉變 | 3個大單區 | 大戶洗盤 | 🟡平衡區
🔴$0.012500-$0.013200 🟢$0.008900-$0.009600
📊 評分明細: 技術突破: 25分 | 趨勢轉變: 20分 | 大單區: 15分

🥈 2. UNFI_USDT 🚀 72分
💰 $0.224800 | 📊 向上突破 | 🏦 7.83%
🎯 突破確認 | 5個大單區 | 2個價格缺口 | 🔴高價區
🔴$0.280000-$0.295000 🟢$0.180000-$0.195000
📊 評分明細: 技術突破: 25分 | 突破確認: 15分 | 大單區: 15分
```

## 🎨 進階功能

### SMC 價格區間分析
- **動態計算**：基於最新 100 根 K 線數據
- **實時更新**：每次分析使用最新市場數據
- **精確定位**：提供具體價格範圍而非模糊描述

### 多重訊號確認
- **Vegas 主導**：傳統 Vegas 通道作為主要訊號
- **SMC 增強**：SMC 指標提供額外確認
- **純 SMC 模式**：當 Vegas 無訊號時，檢測純 SMC 機會

### 智能評分系統
評分考慮多個維度：
- 技術訊號強度
- 市場結構變化
- 機構行為指標
- 流動性狀況
- 年化利率吸引力

## 📊 技術指標說明

### Vegas 通道訊號
- **🚀 向上突破 (LONG_BREAKOUT)**: 價格突破 Vegas 通道上軌
- **⬆️ 向上反彈 (LONG_BOUNCE)**: 價格在上軌獲得支撐反彈
- **📉 向下跌破 (SHORT_BREAKDOWN)**: 價格跌破 Vegas 通道下軌
- **⬇️ 失敗反彈 (SHORT_FAILED_BOUNCE)**: 下軌反彈失敗

### SMC 關鍵概念
- **BOS (Break of Structure)**: 市場結構突破，趨勢延續訊號
- **CHoCH (Change of Character)**: 趨勢性格改變，轉勢訊號
- **Order Block**: 機構大單留下的價格區域，常成為強支撐/阻力
- **Fair Value Gap**: 價格快速移動留下的缺口，具有磁性效應
- **Liquidity Sweep**: 機構掃蕩散戶止損後反向操作

## ⚠️ 風險提示與免責聲明

- **教育目的**：本專案僅供學習和技術分析參考
- **非投資建議**：所有訊號和分析不構成投資建議
- **市場風險**：加密貨幣市場波動巨大，請謹慎投資
- **自主判斷**：使用者應結合自身情況做出獨立判斷
- **API 限制**：請遵守 Gate.io API 使用條款和頻率限制

## 🔧 開發者資訊

### 自定義配置
您可以在 `smc_config.py` 中調整：
- SMC 指標參數
- 評分權重
- 顯示閾值
- 時間週期設定

### 擴展開發
專案採用模組化設計，易於擴展：
- 添加新的技術指標
- 自定義評分邏輯
- 整合其他交易所 API
- 實現多時間框架分析

## 📞 支援與貢獻

如有問題或建議，歡迎：
- 提交 Issue
- 發起 Pull Request
- 參與討論和改進

---

**⚡ Powered by Vegas Channel & Smart Money Concepts**  
**🚀 專業級加密貨幣交易訊號系統**