# AgenticRAG_ESG

ESG Agent 是一個基於 Llamaindex 的應用程序，用於分析和查詢多個公司的 ESG（環境、社會和治理）報告。該應用程序使用先進的 AI 代理來處理用戶查詢，並提供相關的 ESG 信息。

## 功能特點

- 上傳和處理多個公司的 ESG 報告 PDF 文件
- 使用 AI 代理分析 ESG 報告內容
- 回答有關特定公司或整個行業的 ESG 相關問題
- 提供公司的行業分類和相關備註信息
- 實時更新公司列表和處理新上傳的文件
- 高效的文檔處理和索引創建

## 項目結構

- `main.py`: 主應用程序文件，包含 Streamlit 界面和主要邏輯
- `Config`: 配置類，包含重要的設置和路徑
- `SettingsManager`: 管理 LlamaIndex 設置
- `AgentBuilder`: 構建各種 AI 代理
- `ToolManager`: 管理和組織查詢工具
- `ESGAgent`: 主要的 ESG 分析代理
- `data_processing.py`: 包含 `DocumentLoader` 類，負責加載和處理文檔
- `indexing.py`: 包含 `IndexBuilder` 類，負責創建和管理文檔索引

## 注意事項

- 確保您有足夠的 OpenAI API 額度來處理查詢。
- 處理大型 PDF 文件可能需要一些時間，請耐心等待。
- 該應用程序使用 GPT-4o 模型，確保您的 OpenAI 賬戶有權限使用該模型。
- 索引創建過程可能佔用大量計算資源，請確保您的系統有足夠的處理能力。
