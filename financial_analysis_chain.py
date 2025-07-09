"""
financial_analysis_chain.py

- Loads API keys from environment variables (supports .env for local dev)
- Initializes OpenAI and Gemini clients (latest SDKs, June 2025)
- Configures DSPy global LM for both OpenAI and Gemini
- Implements the financial analysis prompt chain as a DSPy module/class
- Generates markdown reports in reports/ directory
"""

import os
import sys
import argparse
from datetime import datetime
import yfinance as yf
import pandas as pd
from typing import Dict, Any, Optional
import requests

# Load environment variables from .env if available (for local development)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional; skip if not installed

# --- Load API Keys from Environment Variables ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY not set in environment variables.")
if not GEMINI_API_KEY:
    raise EnvironmentError("GEMINI_API_KEY not set in environment variables.")

# --- Perplexity API Key ---
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
if not PERPLEXITY_API_KEY:
    raise EnvironmentError("PERPLEXITY_API_KEY not set in environment variables.")

PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"

def fetch_perplexity(query: str, section: str) -> str:
    """
    Query Perplexity API for a given section using the latest Perplexity API.
    Always uses model: sonar-pro.
    """
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }
    # Always use the "sonar-pro" model as per latest requirements
    model_id = "sonar-pro"
    data = {
        "model": model_id,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a financial analysis assistant. Answer with detailed, data-driven insights and cite sources where possible."
                )
            },
            {
                "role": "user",
                "content": query
            }
        ],
        "max_tokens": 1024,
        "temperature": 0.2
    }
    try:
        resp = requests.post(PERPLEXITY_API_URL, headers=headers, json=data, timeout=30)
        resp.raise_for_status()
        result = resp.json()
        return result["choices"][0]["message"]["content"].strip()
    except Exception:
        return f"Could not fetch {section} from Perplexity. Please try again later."

# --- Initialize OpenAI Client (latest SDK, June 2025) ---
try:
    import openai
    openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
except ImportError:
    openai_client = None
    print("Warning: openai package not installed.")

# --- Initialize Gemini Client (google.generativeai, June 2025) ---
try:
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_client = genai
except ImportError:
    gemini_client = None
    print("Warning: google-generativeai package not installed.")

# --- Initialize DSPy and Configure Global LM ---
try:
    import dspy
    # Configure global LM - all modules will use this
    dspy.configure(
        lm=dspy.LM("openai/gpt-4o-mini", api_key=OPENAI_API_KEY)
    )
except ImportError:
    print("Warning: dspy package not installed.")

# --- DSPy Model Selectors ---
def get_openai_lm():
    import dspy
    return dspy.LM("openai/gpt-4o-mini", api_key=OPENAI_API_KEY)

def get_gemini_lm():
    import dspy
    # Use a supported Gemini model format
    return dspy.LM("gemini/gemini-1.5-pro", api_key=GEMINI_API_KEY)


# --- Financial Data Fetching Functions ---
def fetch_stock_data(ticker: str) -> Dict[str, Any]:
    """Fetch comprehensive stock data using yfinance."""
    try:
        stock = yf.Ticker(ticker)
        
        # Get basic info
        info = stock.info
        
        # Get financial statements
        financials = stock.financials
        balance_sheet = stock.balance_sheet
        cash_flow = stock.cashflow
        
        # Get historical data (1 year)
        hist = stock.history(period="1y")
        
        # Get recent price data
        current_price = info.get('currentPrice', info.get('regularMarketPrice', 'N/A'))
        
        # Calculate basic metrics
        price_change_1y = ((current_price - hist['Close'].iloc[0]) / hist['Close'].iloc[0] * 100) if len(hist) > 0 and current_price != 'N/A' else 'N/A'
        
        data = {
            'ticker': ticker.upper(),
            'company_name': info.get('longName', ticker.upper()),
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown'),
            'current_price': current_price,
            'market_cap': info.get('marketCap', 'N/A'),
            'pe_ratio': info.get('trailingPE', 'N/A'),
            'forward_pe': info.get('forwardPE', 'N/A'),
            'price_to_book': info.get('priceToBook', 'N/A'),
            'debt_to_equity': info.get('debtToEquity', 'N/A'),
            'dividend_yield': info.get('dividendYield', 'N/A'),
            'profit_margin': info.get('profitMargins', 'N/A'),
            'revenue_growth': info.get('revenueGrowth', 'N/A'),
            'earnings_growth': info.get('earningsGrowth', 'N/A'),
            'price_change_1y': price_change_1y,
            '52_week_high': info.get('fiftyTwoWeekHigh', 'N/A'),
            '52_week_low': info.get('fiftyTwoWeekLow', 'N/A'),
            'avg_volume': info.get('averageVolume', 'N/A'),
            'beta': info.get('beta', 'N/A'),
            'recommendation': info.get('recommendationKey', 'N/A'),
            'target_price': info.get('targetMeanPrice', 'N/A'),
            'analyst_count': info.get('numberOfAnalystOpinions', 'N/A'),
            'business_summary': info.get('businessSummary', 'N/A'),
            'financials': financials,
            'balance_sheet': balance_sheet,
            'cash_flow': cash_flow,
            'historical_data': hist
        }
        
        return data
        
    except Exception as e:
        print(f"Warning: Could not fetch data for {ticker}: {e}")
        return {
            'ticker': ticker.upper(),
            'company_name': f"{ticker.upper()} Corporation",
            'sector': 'Unknown',
            'industry': 'Unknown',
            'error': str(e)
        }

def format_financial_data(data: Dict[str, Any]) -> str:
    """Format financial data for inclusion in DSPy prompts."""
    if 'error' in data:
        return f"Limited data available for {data['ticker']} due to: {data['error']}"
    
    # Format basic metrics
    formatted = f"""
FINANCIAL DATA FOR {data['ticker']} ({data['company_name']}):

BASIC INFORMATION:
- Sector: {data['sector']}
- Industry: {data['industry']}
- Current Price: ${data['current_price']} 
- Market Cap: {format_large_number(data['market_cap'])}
- 52-Week Range: ${data['52_week_low']} - ${data['52_week_high']}
- 1-Year Price Change: {format_percentage(data['price_change_1y'])}

VALUATION METRICS:
- P/E Ratio: {data['pe_ratio']}
- Forward P/E: {data['forward_pe']}
- Price-to-Book: {data['price_to_book']}
- Beta: {data['beta']}

FINANCIAL HEALTH:
- Debt-to-Equity: {data['debt_to_equity']}
- Profit Margin: {format_percentage(data['profit_margin'])}
- Revenue Growth: {format_percentage(data['revenue_growth'])}
- Earnings Growth: {format_percentage(data['earnings_growth'])}
- Dividend Yield: {format_percentage(data['dividend_yield'])}

ANALYST DATA:
- Recommendation: {data['recommendation']}
- Target Price: ${data['target_price']}
- Number of Analysts: {data['analyst_count']}

BUSINESS SUMMARY:
{data['business_summary'][:500] + '...' if data['business_summary'] and len(str(data['business_summary'])) > 500 else data['business_summary']}
"""
    
    return formatted.strip()

def format_large_number(value) -> str:
    """Format large numbers in billions/millions."""
    if value == 'N/A' or value is None:
        return 'N/A'
    try:
        num = float(value)
        if num >= 1e12:
            return f"${num/1e12:.2f}T"
        elif num >= 1e9:
            return f"${num/1e9:.2f}B"
        elif num >= 1e6:
            return f"${num/1e6:.2f}M"
        else:
            return f"${num:,.0f}"
    except:
        return str(value)

def format_percentage(value) -> str:
    """Format percentage values."""
    if value == 'N/A' or value is None:
        return 'N/A'
    try:
        return f"{float(value)*100:.2f}%"
    except:
        return str(value)

def get_recent_financial_statements(data: Dict[str, Any]) -> str:
    """Extract and format recent financial statement data."""
    if 'error' in data:
        return "Financial statements not available."
    
    try:
        financials = data.get('financials')
        balance_sheet = data.get('balance_sheet')
        
        if financials is not None and not financials.empty:
            # Get most recent year data
            recent_col = financials.columns[0]
            revenue = financials.loc['Total Revenue', recent_col] if 'Total Revenue' in financials.index else 'N/A'
            net_income = financials.loc['Net Income', recent_col] if 'Net Income' in financials.index else 'N/A'
            
            statement_data = f"""
RECENT FINANCIAL STATEMENTS (Most Recent Year):
- Total Revenue: {format_large_number(revenue)}
- Net Income: {format_large_number(net_income)}
"""
            
            if balance_sheet is not None and not balance_sheet.empty:
                recent_bs_col = balance_sheet.columns[0]
                total_assets = balance_sheet.loc['Total Assets', recent_bs_col] if 'Total Assets' in balance_sheet.index else 'N/A'
                total_debt = balance_sheet.loc['Total Debt', recent_bs_col] if 'Total Debt' in balance_sheet.index else 'N/A'
                
                statement_data += f"- Total Assets: {format_large_number(total_assets)}\n"
                statement_data += f"- Total Debt: {format_large_number(total_debt)}\n"
            
            return statement_data.strip()
        
    except Exception as e:
        return f"Error processing financial statements: {e}"
    
    return "Financial statements not available."


# --- Financial Analysis Prompt Chain Implementation ---
try:
    import dspy

    import time
    class FinancialAnalysisChain(dspy.Module):
        """
        DSPy module for a 14-step financial analysis prompt chain with dynamic model selection.
        Each step is a method/submodule, using Gemini for retrieval, OpenAI for analysis, or hybrid for both.
        Context (TICKER, COMPANY, SECTOR, etc.) is passed between steps.

        Refactored July 2025: CLI step-by-step progress output for all 14 steps.
        """

        STEP_PROGRESS = [
            ("Company Overview", "OpenAI/Gemini"),
            ("Industry Analysis", "Perplexity"),
            ("Competitor Identification", "OpenAI/Gemini"),
            ("Financial Statement Summary", "yfinance + OpenAI/Gemini"),
            ("Ratio Analysis", "yfinance + OpenAI/Gemini"),
            ("Trend Analysis", "Perplexity"),
            ("SWOT Analysis", "OpenAI/Gemini"),
            ("Management & Governance", "OpenAI/Gemini"),
            ("Analyst Opinions", "Perplexity"),
            ("ESG Factors", "Perplexity"),
            ("Valuation", "OpenAI/Gemini"),
            ("Investment Risks", "OpenAI/Gemini"),
            ("Investment Thesis", "OpenAI/Gemini"),
            ("Summary & Recommendation", "OpenAI/Gemini"),
        ]

        def _print_progress(self, progress_states):
            """
            Prints the checklist progress to the CLI.
            progress_states: list of ("pending"|"done") for each step.
            """
            # Move cursor up to overwrite previous progress if not first print
            if hasattr(self, "_last_progress_lines"):
                sys.stdout.write(f"\033[{self._last_progress_lines}A")
            lines = []
            for idx, (desc, api) in enumerate(self.STEP_PROGRESS):
                state = progress_states[idx]
                if state == "done":
                    icon = "✅"
                else:
                    icon = "⏳"
                lines.append(f"{icon} {desc} ({api})")
            output = "\n".join(lines)
            print(output)
            sys.stdout.flush()
            self._last_progress_lines = len(lines)

        def _init_progress(self):
            self._progress_states = ["pending"] * len(self.STEP_PROGRESS)
            self._print_progress(self._progress_states)

        def _mark_step_done(self, step_idx):
            self._progress_states[step_idx] = "done"
            self._print_progress(self._progress_states)

        def __init__(self):
            super().__init__()
            # Use global LM configuration - no per-module LM specification
            
            # 2. Industry/Sector Overview
            self.industry_overview = dspy.Predict(
                "SECTOR -> industry_overview"
            )

            # 3. Competitor Identification
            self.competitor_search = dspy.Predict(
                "TICKER, COMPANY, SECTOR -> competitors"
            )

            # 4. Financial Statement Summary
            self.financial_summary = dspy.Predict(
                "TICKER, COMPANY -> financial_summary"
            )

            # 5. Ratio Analysis
            self.ratio_analysis = dspy.Predict(
                "financial_summary -> ratio_analysis"
            )

            # 6. Trend Analysis
            self.trend_analysis = dspy.Predict(
                "financial_summary -> trend_analysis"
            )

            # 7. SWOT Analysis
            self.swot_analysis = dspy.Predict(
                "company_overview, industry_overview, competitors, ratio_analysis, trend_analysis -> swot"
            )

            # 8. Management & Governance
            self.management_governance = dspy.Predict(
                "TICKER, COMPANY -> management_governance"
            )

            # 9. Analyst Opinions
            self.analyst_opinions = dspy.Predict(
                "TICKER, COMPANY -> analyst_opinions"
            )

            # 10. ESG Factors
            self.esg_factors = dspy.Predict(
                "TICKER, COMPANY -> esg_factors"
            )

            # 11. Valuation
            self.valuation = dspy.Predict(
                "financial_summary, ratio_analysis, trend_analysis, analyst_opinions -> valuation"
            )

            # 12. Investment Risks
            self.risk_retrieval = dspy.Predict(
                "TICKER, COMPANY -> risk_factors_raw"
            )
            self.risk_synthesis = dspy.Predict(
                "risk_factors_raw, swot -> investment_risks"
            )

            # 13. Investment Thesis
            self.investment_thesis = dspy.Predict(
                "swot, valuation, investment_risks -> thesis"
            )

            # 14. Summary & Recommendation
            self.summary_recommendation = dspy.Predict(
                "thesis, valuation, analyst_opinions -> summary_recommendation"
            )

        def forward(self, TICKER, COMPANY, SECTOR):
            """
            Run the full 14-step financial analysis chain with real financial data.
            Returns a dict with all intermediate and final outputs.
            """
            print(f"📊 Fetching real-time financial data for {TICKER}...")

            # Initialize progress checklist
            self._init_progress()

            # Fetch real financial data first (not a checklist step)
            stock_data = fetch_stock_data(TICKER)
            financial_data_str = format_financial_data(stock_data)
            financial_statements_str = get_recent_financial_statements(stock_data)

            # Use real company data if available
            if 'error' not in stock_data:
                COMPANY = stock_data.get('company_name', COMPANY)
                SECTOR = stock_data.get('sector', SECTOR)

            context = {
                "TICKER": TICKER,
                "COMPANY": COMPANY,
                "SECTOR": SECTOR,
                "real_financial_data": stock_data,
                "financial_data_summary": financial_data_str
            }

            # 1. Company Overview (OpenAI/Gemini)
            step_idx = 0
            self._print_progress(self._progress_states)
            company_overview_predict = dspy.Predict("financial_data, company_info -> company_overview")
            company_overview_result = company_overview_predict(
                financial_data=financial_data_str,
                company_info=f"Provide a comprehensive overview of {COMPANY} ({TICKER}) based on the real financial data provided."
            )
            company_overview = getattr(company_overview_result, "company_overview", company_overview_result)
            context["company_overview"] = company_overview
            self._mark_step_done(step_idx)

            # 2. Industry Analysis (Perplexity)
            step_idx = 1
            industry_query = f"Industry and sector overview for {COMPANY} ({TICKER}) in the {SECTOR} sector. Include recent trends and market context."
            self._print_progress(self._progress_states)
            industry_overview = fetch_perplexity(industry_query, "industry overview")
            context["industry_overview"] = industry_overview
            self._mark_step_done(step_idx)

            # 3. Competitor Identification (OpenAI/Gemini)
            step_idx = 2
            self._print_progress(self._progress_states)
            competitors = self.competitor_search(TICKER=TICKER, COMPANY=COMPANY, SECTOR=SECTOR).competitors
            context["competitors"] = competitors
            self._mark_step_done(step_idx)

            # 4. Financial Statement Summary (yfinance + OpenAI/Gemini)
            step_idx = 3
            self._print_progress(self._progress_states)
            financial_summary = self.financial_summary(
                TICKER=TICKER,
                COMPANY=f"{COMPANY} with real financial data: {financial_statements_str}"
            ).financial_summary
            context["financial_summary"] = financial_summary
            self._mark_step_done(step_idx)

            # 5. Ratio Analysis (yfinance + OpenAI/Gemini)
            step_idx = 4
            self._print_progress(self._progress_states)
            ratio_analysis = self.ratio_analysis(
                financial_summary=f"Real financial metrics for {TICKER}: {financial_data_str}\n\nDetailed analysis: {financial_summary}"
            ).ratio_analysis
            context["ratio_analysis"] = ratio_analysis
            self._mark_step_done(step_idx)

            # 6. Trend Analysis (Perplexity)
            step_idx = 5
            self._print_progress(self._progress_states)
            trend_query = f"Recent price and financial trends for {COMPANY} ({TICKER}) in the {SECTOR} sector. 1-Year Price Change: {stock_data.get('price_change_1y', 'N/A')}, Current: ${stock_data.get('current_price', 'N/A')}, 52W Range: ${stock_data.get('52_week_low', 'N/A')}-${stock_data.get('52_week_high', 'N/A')}."
            trend_analysis = fetch_perplexity(trend_query, "trend analysis")
            context["trend_analysis"] = trend_analysis
            self._mark_step_done(step_idx)

            # 7. SWOT Analysis (OpenAI/Gemini)
            step_idx = 6
            self._print_progress(self._progress_states)
            swot = self.swot_analysis(
                company_overview=company_overview,
                industry_overview=industry_overview,
                competitors=competitors,
                ratio_analysis=ratio_analysis,
                trend_analysis=trend_analysis
            ).swot
            context["swot"] = swot
            self._mark_step_done(step_idx)

            # 8. Management & Governance (OpenAI/Gemini)
            step_idx = 7
            self._print_progress(self._progress_states)
            management_governance = self.management_governance(TICKER=TICKER, COMPANY=COMPANY).management_governance
            context["management_governance"] = management_governance
            self._mark_step_done(step_idx)

            # 9. Analyst Opinions (Perplexity)
            step_idx = 8
            self._print_progress(self._progress_states)
            analyst_query = f"Summarize the latest analyst opinions, price targets, and recommendations for {COMPANY} ({TICKER}) in the {SECTOR} sector."
            analyst_opinions = fetch_perplexity(analyst_query, "analyst opinions")
            context["analyst_opinions"] = analyst_opinions
            self._mark_step_done(step_idx)

            # 10. ESG Factors (Perplexity)
            step_idx = 9
            self._print_progress(self._progress_states)
            esg_query = f"Summarize the latest ESG (Environmental, Social, Governance) factors and news for {COMPANY} ({TICKER})."
            esg_factors = fetch_perplexity(esg_query, "ESG factors")
            context["esg_factors"] = esg_factors
            self._mark_step_done(step_idx)

            # 11. Valuation (OpenAI/Gemini)
            step_idx = 10
            self._print_progress(self._progress_states)
            real_valuation_metrics = f"Current valuation metrics: P/E: {stock_data.get('pe_ratio', 'N/A')}, Forward P/E: {stock_data.get('forward_pe', 'N/A')}, P/B: {stock_data.get('price_to_book', 'N/A')}, Current Price: ${stock_data.get('current_price', 'N/A')}"
            valuation = self.valuation(
                financial_summary=financial_summary,
                ratio_analysis=f"{ratio_analysis}\n\nReal valuation data: {real_valuation_metrics}",
                trend_analysis=trend_analysis,
                analyst_opinions=f"{analyst_opinions}\n\nReal analyst target: ${stock_data.get('target_price', 'N/A')}"
            ).valuation
            context["valuation"] = valuation
            self._mark_step_done(step_idx)

            # 12. Investment Risks (OpenAI/Gemini)
            step_idx = 11
            self._print_progress(self._progress_states)
            risk_factors_raw = self.risk_retrieval(TICKER=TICKER, COMPANY=COMPANY).risk_factors_raw
            investment_risks = self.risk_synthesis(risk_factors_raw=risk_factors_raw, swot=swot).investment_risks
            context["investment_risks"] = investment_risks
            self._mark_step_done(step_idx)

            # 13. Investment Thesis (OpenAI/Gemini)
            step_idx = 12
            self._print_progress(self._progress_states)
            thesis = self.investment_thesis(
                swot=swot,
                valuation=valuation,
                investment_risks=investment_risks
            ).thesis
            context["thesis"] = thesis
            self._mark_step_done(step_idx)

            # 14. Summary & Recommendation (OpenAI/Gemini)
            step_idx = 13
            self._print_progress(self._progress_states)
            summary_recommendation = self.summary_recommendation(
                thesis=thesis,
                valuation=valuation,
                analyst_opinions=analyst_opinions
            ).summary_recommendation
            context["summary_recommendation"] = summary_recommendation
            self._mark_step_done(step_idx)

            # Print summary
            print("\n🎉 All 14 analysis steps complete!\n")

            return context

except ImportError:
    # If dspy is not installed, keep the placeholder class
    class FinancialAnalysisChain:
        def __init__(self):
            pass
        def run(self, input_data):
            raise NotImplementedError("DSPy not installed; financial analysis logic not available.")

def generate_markdown_report(ticker, company, sector, results):
    """Generate a markdown report from the analysis results."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""# Financial Analysis Report: {company} ({ticker})

**Sector:** {sector}  
**Generated:** {timestamp}  
**Analysis Method:** DSPy 14-Step Financial Analysis Chain

---

## Executive Summary

{results.get('summary_recommendation', 'No summary available.')}

---

## Company Overview

{results.get('company_overview', 'No company overview available.')}

---

## Industry Analysis

{results.get('industry_overview', 'No industry analysis available.')}

---

## Competitive Landscape

**Key Competitors:** {results.get('competitors', 'No competitor data available.')}

---

## Financial Analysis

### Financial Summary
{results.get('financial_summary', 'No financial summary available.')}

### Financial Ratios
{results.get('ratio_analysis', 'No ratio analysis available.')}

### Trend Analysis
{results.get('trend_analysis', 'No trend analysis available.')}

---

## Strategic Analysis

### SWOT Analysis
{results.get('swot', 'No SWOT analysis available.')}

### Management & Governance
{results.get('management_governance', 'No management analysis available.')}

---

## Investment Analysis

### Valuation
{results.get('valuation', 'No valuation analysis available.')}

### Investment Risks
{results.get('investment_risks', 'No risk analysis available.')}

### Investment Thesis
{results.get('thesis', 'No investment thesis available.')}

---

## Additional Factors

### Analyst Opinions
{results.get('analyst_opinions', 'No analyst opinions available.')}

### ESG Factors
{results.get('esg_factors', 'No ESG analysis available.')}

---

*Report generated using DSPy Financial Analysis Chain*
"""
    return report

def save_report(ticker, report_content):
    """Save the report to the reports directory."""
    import os
    
    # Create reports directory if it doesn't exist
    reports_dir = "reports"
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)
    
    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{ticker}_{timestamp}_analysis.md"
    filepath = os.path.join(reports_dir, filename)
    
    # Write report
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    return filepath

# Company mapping - you can expand this as needed
COMPANY_INFO = {
    "NVDA": {"company": "NVIDIA Corporation", "sector": "Semiconductors"},
    "AAPL": {"company": "Apple Inc.", "sector": "Technology"},
    "MSFT": {"company": "Microsoft Corporation", "sector": "Technology"},
    "GOOGL": {"company": "Alphabet Inc.", "sector": "Technology"},
    "TSLA": {"company": "Tesla Inc.", "sector": "Automotive"},
    "META": {"company": "Meta Platforms Inc.", "sector": "Technology"},
    "AMZN": {"company": "Amazon.com Inc.", "sector": "E-commerce"},
}

def get_company_info(ticker):
    """Get company name and sector for a ticker."""
    ticker = ticker.upper()
    if ticker in COMPANY_INFO:
        return COMPANY_INFO[ticker]["company"], COMPANY_INFO[ticker]["sector"]
    else:
        # Default fallback
        return f"{ticker} Corporation", "Unknown"

# Module initialization message
if __name__ != "__main__":
    print("financial_analysis_chain.py initialized. FinancialAnalysisChain ready for use.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate financial analysis report for a stock ticker')
    parser.add_argument('ticker', help='Stock ticker symbol (e.g., NVDA, AAPL, MSFT, TSLA)')
    
    args = parser.parse_args()
    
    # Get company info from ticker
    TICKER = args.ticker.upper()
    COMPANY, SECTOR = get_company_info(TICKER)
    
    print(f"Generating financial analysis for {COMPANY} ({TICKER}) in {SECTOR} sector...")
    
    try:
        import dspy
        chain = FinancialAnalysisChain()
        results = chain.forward(TICKER=TICKER, COMPANY=COMPANY, SECTOR=SECTOR)
        
        # Generate and save report
        report_content = generate_markdown_report(TICKER, COMPANY, SECTOR, results)
        report_path = save_report(TICKER, report_content)
        
        print(f"\n✅ Analysis complete! Report saved to: {report_path}")
        print(f"\n=== Executive Summary ===")
        print(results.get("summary_recommendation", "No summary generated."))
        
    except Exception as e:
        print(f"❌ Error running FinancialAnalysisChain: {e}")