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


# --- Financial Analysis Prompt Chain Implementation ---
try:
    import dspy

    class FinancialAnalysisChain(dspy.Module):
        """
        DSPy module for a 14-step financial analysis prompt chain with dynamic model selection.
        Each step is a method/submodule, using Gemini for retrieval, OpenAI for analysis, or hybrid for both.
        Context (TICKER, COMPANY, SECTOR, etc.) is passed between steps.
        """

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
            Run the full 14-step financial analysis chain.
            Returns a dict with all intermediate and final outputs.
            """
            context = {
                "TICKER": TICKER,
                "COMPANY": COMPANY,
                "SECTOR": SECTOR
            }

            # 1. Company Overview
            # Instantiate Predict module inside the method
            company_overview_predict = dspy.Predict("question: str -> answer: str")
            company_overview_result = company_overview_predict(question=f"Provide a brief overview of {COMPANY} ({TICKER}) in the {SECTOR} sector.")
            company_overview = getattr(company_overview_result, "answer", company_overview_result)
            context["company_overview"] = company_overview

            # 2. Industry/Sector Overview
            industry_overview = self.industry_overview(SECTOR=SECTOR).industry_overview
            context["industry_overview"] = industry_overview

            # 3. Competitor Identification
            competitors = self.competitor_search(TICKER=TICKER, COMPANY=COMPANY, SECTOR=SECTOR).competitors
            context["competitors"] = competitors

            # 4. Financial Statement Summary
            financial_summary = self.financial_summary(TICKER=TICKER, COMPANY=COMPANY).financial_summary
            context["financial_summary"] = financial_summary

            # 5. Ratio Analysis
            ratio_analysis = self.ratio_analysis(financial_summary=financial_summary).ratio_analysis
            context["ratio_analysis"] = ratio_analysis

            # 6. Trend Analysis
            trend_analysis = self.trend_analysis(financial_summary=financial_summary).trend_analysis
            context["trend_analysis"] = trend_analysis

            # 7. SWOT Analysis
            swot = self.swot_analysis(
                company_overview=company_overview,
                industry_overview=industry_overview,
                competitors=competitors,
                ratio_analysis=ratio_analysis,
                trend_analysis=trend_analysis
            ).swot
            context["swot"] = swot

            # 8. Management & Governance
            management_governance = self.management_governance(TICKER=TICKER, COMPANY=COMPANY).management_governance
            context["management_governance"] = management_governance

            # 9. Analyst Opinions
            analyst_opinions = self.analyst_opinions(TICKER=TICKER, COMPANY=COMPANY).analyst_opinions
            context["analyst_opinions"] = analyst_opinions

            # 10. ESG Factors
            esg_factors = self.esg_factors(TICKER=TICKER, COMPANY=COMPANY).esg_factors
            context["esg_factors"] = esg_factors

            # 11. Valuation
            valuation = self.valuation(
                financial_summary=financial_summary,
                ratio_analysis=ratio_analysis,
                trend_analysis=trend_analysis,
                analyst_opinions=analyst_opinions
            ).valuation
            context["valuation"] = valuation

            # 12. Investment Risks (Hybrid)
            risk_factors_raw = self.risk_retrieval(TICKER=TICKER, COMPANY=COMPANY).risk_factors_raw
            investment_risks = self.risk_synthesis(risk_factors_raw=risk_factors_raw, swot=swot).investment_risks
            context["investment_risks"] = investment_risks

            # 13. Investment Thesis
            thesis = self.investment_thesis(
                swot=swot,
                valuation=valuation,
                investment_risks=investment_risks
            ).thesis
            context["thesis"] = thesis

            # 14. Summary & Recommendation
            summary_recommendation = self.summary_recommendation(
                thesis=thesis,
                valuation=valuation,
                analyst_opinions=analyst_opinions
            ).summary_recommendation
            context["summary_recommendation"] = summary_recommendation

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