"""
financial_analysis_chain.py

- Loads API keys from environment variables (supports .env for local dev)
- Initializes OpenAI and Gemini clients (latest SDKs, June 2025)
- Configures DSPy global LM for both OpenAI and Gemini
- Implements the financial analysis prompt chain as a DSPy module/class
"""

import os

# Load environment variables from .env if available (for local development)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional; skip if not installed

# --- Minimal DSPy Predict Test ---
def minimal_dspy_test():
    import dspy
    lm = dspy.LM("openai/gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    dspy.configure(lm=lm)
    qa = dspy.Predict("question: str -> answer: str")
    try:
        result = qa(question="What is 2 + 2?")
        print("[MINIMAL DSPY TEST] answer:", getattr(result, "answer", result))
    except Exception as e:
        print("[MINIMAL DSPY TEST] error:", e)

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
    # Default to OpenAI for global LM (step-level override below)
    dspy.settings.configure(
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
    return dspy.LM("gemini/gemini-2.5-pro-preview-03-25", api_key=GEMINI_API_KEY)
    import dspy
    return dspy.GoogleGemini(api_key=GEMINI_API_KEY, model="gemini-1.5-pro")


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
            # Define submodules for each step with appropriate model selection

            # 1. Company Overview (Gemini: retrieval)
            # self.company_overview = dspy.Predict(
            #     "question: str -> answer: str",
            #     lm=get_openai_lm()
            # )

            # 2. Industry/Sector Overview (Gemini: retrieval)
            self.industry_overview = dspy.Predict(
                "SECTOR -> industry_overview",
                lm=get_openai_lm()
            )

            # 3. Competitor Identification (Gemini: retrieval)
            self.competitor_search = dspy.Predict(
                "TICKER, COMPANY, SECTOR -> competitors",
                lm=get_openai_lm()
            )

            # 4. Financial Statement Summary (OpenAI: generative/analysis)
            self.financial_summary = dspy.Predict(
                "TICKER, COMPANY -> financial_summary",
                lm=get_openai_lm()
            )

            # 5. Ratio Analysis (OpenAI: generative/calculation)
            self.ratio_analysis = dspy.Predict(
                "financial_summary -> ratio_analysis",
                lm=get_openai_lm()
            )

            # 6. Trend Analysis (OpenAI: generative/analysis)
            self.trend_analysis = dspy.Predict(
                "financial_summary -> trend_analysis",
                lm=get_openai_lm()
            )

            # 7. SWOT Analysis (OpenAI: generative)
            self.swot_analysis = dspy.Predict(
                "company_overview, industry_overview, competitors, ratio_analysis, trend_analysis -> swot",
                lm=get_openai_lm()
            )

            # 8. Management & Governance (Gemini: retrieval)
            self.management_governance = dspy.Predict(
                "TICKER, COMPANY -> management_governance",
                lm=get_gemini_lm()
            )

            # 9. Analyst Opinions (Gemini: retrieval)
            self.analyst_opinions = dspy.Predict(
                "TICKER, COMPANY -> analyst_opinions",
                lm=get_gemini_lm()
            )

            # 10. ESG Factors (Gemini: retrieval)
            self.esg_factors = dspy.Predict(
                "TICKER, COMPANY -> esg_factors",
                lm=get_gemini_lm()
            )

            # 11. Valuation (OpenAI: generative/calculation)
            self.valuation = dspy.Predict(
                "financial_summary, ratio_analysis, trend_analysis, analyst_opinions -> valuation",
                lm=get_openai_lm()
            )

            # 12. Investment Risks (Hybrid: Gemini retrieval + OpenAI synthesis)
            self.risk_retrieval = dspy.Predict(
                "TICKER, COMPANY -> risk_factors_raw",
                lm=get_gemini_lm()
            )
            self.risk_synthesis = dspy.Predict(
                "risk_factors_raw, swot -> investment_risks",
                lm=get_openai_lm()
            )

            # 13. Investment Thesis (OpenAI: generative/synthesis)
            self.investment_thesis = dspy.Predict(
                "swot, valuation, investment_risks -> thesis",
                lm=get_openai_lm()
            )

            # 14. Summary & Recommendation (OpenAI: generative/synthesis)
            self.summary_recommendation = dspy.Predict(
                "thesis, valuation, analyst_opinions -> summary_recommendation",
                lm=get_openai_lm()
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
            company_overview_predict = dspy.Predict("question: str -> answer: str", lm=get_openai_lm())
            company_overview_result = company_overview_predict(question=f"Provide a brief overview of {COMPANY} ({TICKER}) in the {SECTOR} sector.")
            company_overview = getattr(company_overview_result, "answer", company_overview_result)
            print(f"[DEBUG] company_overview type: {type(company_overview)}, value: {company_overview}")
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

# Direct test of company_overview with minimal signature
# Minimal Predict test inside a function (not a class)
    def minimal_predict_in_function():
        import dspy
        lm = dspy.LM("openai/gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
        dspy.configure(lm=lm)
        qa = dspy.Predict("question: str -> answer: str")
        try:
            result = qa(question="What is 2 + 2?")
            print("[MINIMAL FUNCTION TEST] answer:", getattr(result, "answer", result))
        except Exception as e:
            print("[MINIMAL FUNCTION TEST] error:", e)
    minimal_predict_in_function()
    try:
        import dspy
        chain = FinancialAnalysisChain()
        result = chain.company_overview(question="Tell me about NVIDIA")
        print("[DIRECT company_overview TEST] answer:", getattr(result, "answer", result))
    except Exception as e:
        print("[DIRECT company_overview TEST] error:", e)
    # Example usage (commented out)
    # chain = FinancialAnalysisChain()
    # result = chain.forward(TICKER="AAPL", COMPANY="Apple Inc.", SECTOR="Technology")

# Run minimal DSPy Predict test
    minimal_dspy_test()
except ImportError:
    # If dspy is not installed, keep the placeholder class
    class FinancialAnalysisChain:
        def __init__(self):
            pass
        def run(self, input_data):
            raise NotImplementedError("DSPy not installed; financial analysis logic not available.")

if __name__ == "__main__":
    print("financial_analysis_chain.py initialized. FinancialAnalysisChain ready for use.")
if __name__ == "__main__":
    # Example run for NVIDIA (NVDA) in the Semiconductors sector
    TICKER = "NVDA"
    COMPANY = "NVIDIA"
    SECTOR = "Semiconductors"

    try:
        import dspy
        chain = FinancialAnalysisChain()
        results = chain.forward(TICKER=TICKER, COMPANY=COMPANY, SECTOR=SECTOR)
        print("\n=== Final Summary & Recommendation ===")
        print(results.get("summary_recommendation", "No summary generated."))
        print("\n=== All Results ===")
        for k, v in results.items():
            print(f"{k}: {v}\n")
    except Exception as e:
        print(f"Error running FinancialAnalysisChain: {e}")
# --- Minimal DSPy Predict Test ---
def minimal_dspy_test():
    import dspy
    lm = dspy.LM("openai/gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    dspy.configure(lm=lm)
    qa = dspy.Predict("question: str -> answer: str")
    try:
        result = qa(question="What is 2 + 2?")
        print("[MINIMAL DSPY TEST] answer:", getattr(result, "answer", result))
    except Exception as e:
        print("[MINIMAL DSPY TEST] error:", e)