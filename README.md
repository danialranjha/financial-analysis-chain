# Financial Analysis Chain

A DSPy-powered financial analysis tool that generates comprehensive investment reports for publicly traded companies using real-time financial data and AI-driven analysis.

## Features

- **Real-Time Financial Data**: Fetches live stock prices, financial statements, ratios, and analyst recommendations using yfinance
- **14-Step Analysis Pipeline**: Comprehensive financial analysis including company overview, industry analysis, financial metrics, SWOT analysis, valuation, and investment recommendations
- **AI-Powered Insights**: Uses OpenAI GPT-4o-mini for analysis and synthesis tasks with real financial data as context
- **Markdown Reports**: Generates professional, structured reports saved to `reports/` directory
- **Simple CLI**: Just provide a stock ticker symbol to get started

## Installation

1. **Clone the repository and navigate to the project directory**

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install dspy openai google-generativeai python-dotenv yfinance
   ```

4. **Set up API keys:**
   - Copy `.env.template` to `.env`
   - Add your OpenAI and Gemini API keys to the `.env` file:
     ```
     OPENAI_API_KEY=your_openai_api_key_here
     GEMINI_API_KEY=your_gemini_api_key_here
     ```

## Usage

### Basic Usage

Generate a financial analysis report for any stock ticker:

```bash
python financial_analysis_chain.py TICKER
```

### Examples

```bash
# Analyze NVIDIA
python financial_analysis_chain.py NVDA

# Analyze Tesla
python financial_analysis_chain.py TSLA

# Analyze Apple
python financial_analysis_chain.py AAPL

# Analyze Meta
python financial_analysis_chain.py META
```

### Help

```bash
python financial_analysis_chain.py --help
```

## Supported Companies

The tool includes built-in mappings for major companies:

| Ticker | Company | Sector |
|--------|---------|---------|
| NVDA | NVIDIA Corporation | Semiconductors |
| AAPL | Apple Inc. | Technology |
| MSFT | Microsoft Corporation | Technology |
| GOOGL | Alphabet Inc. | Technology |
| TSLA | Tesla Inc. | Automotive |
| META | Meta Platforms Inc. | Technology |
| AMZN | Amazon.com Inc. | E-commerce |

For other tickers, the system will attempt analysis using the ticker symbol.

## Real-Time Financial Data

The tool automatically fetches current financial data for each analysis, including:

### Market Data
- Current stock price and market capitalization
- 52-week high/low prices
- 1-year price performance
- Daily trading volume and beta

### Financial Metrics
- P/E ratio (trailing and forward)
- Price-to-book ratio
- Debt-to-equity ratio
- Profit margins and growth rates
- Dividend yield

### Analyst Information
- Consensus recommendation (buy/hold/sell)
- Analyst target price
- Number of analysts covering the stock

### Financial Statements
- Recent revenue and net income
- Total assets and debt levels
- Cash flow information

### Company Information
- Business summary and description
- Industry and sector classification
- Key business metrics

This real-time data is integrated into every step of the analysis to provide current, accurate insights rather than relying solely on AI training data.

## Report Structure

Each generated report includes:

- **Executive Summary** - Investment recommendation and key insights
- **Company Overview** - Business description and market position  
- **Industry Analysis** - Sector overview and market trends
- **Competitive Landscape** - Key competitors identification
- **Financial Analysis** - Financial summary, ratios, and trend analysis
- **Strategic Analysis** - SWOT analysis and management evaluation
- **Investment Analysis** - Valuation, risk assessment, and investment thesis
- **Additional Factors** - Analyst opinions and ESG considerations

## Output

Reports are automatically saved to the `reports/` directory with timestamped filenames:

```
reports/
├── NVDA_20250624_185032_analysis.md
├── TSLA_20250624_185146_analysis.md
└── AAPL_20250624_185642_analysis.md
```

## Example Output

```bash
$ python financial_analysis_chain.py AAPL
Generating financial analysis for Apple Inc. (AAPL) in Technology sector...
📊 Fetching real-time financial data for AAPL...

✅ Analysis complete! Report saved to: reports/AAPL_20250624_231459_analysis.md

=== Executive Summary ===
Given Apple's strong brand loyalty, solid financial health, and the potential for growth 
in emerging technologies, the stock is currently undervalued at $200.3 compared to the 
analyst target price of $228.85. With a consensus recommendation to buy from 40 analysts, 
the investment outlook is positive despite some risks...
```

## Requirements

- Python 3.8+
- OpenAI API key
- Gemini API key
- Internet connection for API calls

## Notes

- **Real-time data**: Analysis uses current financial data fetched from Yahoo Finance via yfinance
- **Data accuracy**: Financial data is pulled from reliable sources but may have slight delays
- Reports are excluded from version control (added to `.gitignore`)
- Each analysis typically takes 1-2 minutes depending on API response times and data fetching
- The tool uses AI models with real financial data, so results may vary between runs
- Generated reports are for informational purposes only and should not be considered as financial advice

## Troubleshooting

**Environment variable errors:**
- Ensure your `.env` file contains valid API keys
- Check that the virtual environment is activated

**Import errors:**
- Verify all dependencies are installed: `pip install -r requirements.txt` (if available)
- Make sure you're using the correct Python environment

**API errors:**
- Verify your API keys are valid and have sufficient quota
- Check your internet connection