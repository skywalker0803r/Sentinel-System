# Sentinel-System - Vegas 通道加密貨幣交易訊號機器人

## 專案簡介

Sentinel-System 是一個基於 Python 開發的 Discord 機器人，旨在為加密貨幣交易者提供自動化的「Vegas 通道」交易訊號分析與通知服務。它透過連接 Gate.io 交易所的 API，獲取實時的市場數據，並運用 Vegas 通道策略識別潛在的交易機會，隨後將分析結果發送到指定的 Discord 頻道。

## 功能特色

*   **Gate.io 數據整合**：自動從 Gate.io API 獲取加密貨幣的 K 線數據和借貸年化利率 (APR)。
*   **Vegas 通道策略分析**：
    *   計算多個指數移動平均線 (EMA)，包括 EMA12, EMA144, EMA169, EMA576, EMA676。
    *   識別四種關鍵的 Vegas 通道轉折訊號：
        *   `LONG_BREAKOUT` (多頭突破)
        *   `LONG_BOUNCE` (多頭反彈)
        *   `SHORT_BREAKDOWN` (空頭跌破)
        *   `SHORT_FAILED_BOUNCE` (空頭反彈失敗)
*   **Discord 實時通知**：
    *   自動將分析出的 Top 5 多頭和 Top 5 空頭訊號（依 APR 排序）發送到指定的 Discord 頻道。
    *   訊息包含交易對、收盤價、訊號類型和年化利率等詳細資訊。
*   **自動化運行**：機器人啟動後會自動執行數據獲取、分析和通知流程。

## 安裝與設定

### 1. 環境準備

確保您的系統已安裝 Python 3.8 或更高版本。

### 2. 安裝依賴

使用 `requirements.txt` 安裝所有必要的 Python 函式庫：

```bash
pip install -r requirements.txt
```

### 3. Discord 機器人設定

1.  **創建 Discord 應用程式與機器人**：
    *   前往 [Discord Developer Portal](https://discord.com/developers/applications)。
    *   點擊 `New Application` 創建一個新的應用程式。
    *   在應用程式頁面左側導航欄中選擇 `Bot`，然後點擊 `Add Bot`。
    *   啟用 `Message Content Intent` (在 `Privileged Gateway Intents` 下)。
    *   複製您的機器人 `TOKEN`。
2.  **獲取頻道 ID**：
    *   在您的 Discord 伺服器中，開啟開發者模式 (使用者設定 -> 進階 -> 開發者模式)。
    *   右鍵點擊您希望機器人發送訊息的頻道，選擇 `複製 ID`。

### 4. 配置 `bot.py`

打開 `bot.py` 檔案，並將以下變數替換為您自己的值：

```python
DISCORD_TOKEN = "您的 Discord 機器人 TOKEN"  # 替換為您複製的機器人 TOKEN
CHANNEL_ID = 您的頻道 ID  # 替換為您複製的 Discord 頻道 ID
```

### 5. 運行機器人

在專案根目錄下運行 `bot.py`：

```bash
python bot.py
```

機器人啟動後，將會自動連接到 Discord 並開始發送 Vegas 通道訊號。

## 使用說明

機器人啟動後，會自動在指定的 Discord 頻道發送分析結果。您無需手動輸入任何指令。它會定期（根據程式碼邏輯，目前是在啟動時執行一次 `send_vegas_signals`）發送最新的訊號。

## 注意事項

*   本專案僅供學習和參考，不構成任何投資建議。加密貨幣市場波動性大，請謹慎投資。
*   請妥善保管您的 Discord 機器人 TOKEN，切勿洩露。
*   Gate.io API 的使用可能會有頻率限制，請注意遵守其使用條款。