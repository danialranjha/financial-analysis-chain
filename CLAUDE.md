# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a DSPy-based financial analysis system that implements a 14-step prompt chain for comprehensive stock analysis. The codebase contains two main files:

- `financial_analysis_chain.py`: Full implementation with hybrid model selection (OpenAI + Gemini)
- `dspy_prompt_chain_demo.py`: Simple demo showing basic DSPy chaining concepts

## Architecture

### Core Components

**FinancialAnalysisChain (DSPy Module)**
- 14-step analysis pipeline implemented as DSPy submodules
- Dynamic model selection: Gemini for retrieval tasks, OpenAI for analysis/generation
- Context passing between steps (TICKER, COMPANY, SECTOR, plus intermediate results)
- Each step is a `dspy.Predict` module with specific input/output signatures

**Analysis Steps**:
1. Company Overview → 2. Industry Overview → 3. Competitor Analysis → 4. Financial Summary → 5. Ratio Analysis → 6. Trend Analysis → 7. SWOT Analysis → 8. Management & Governance → 9. Analyst Opinions → 10. ESG Factors → 11. Valuation → 12. Investment Risks → 13. Investment Thesis → 14. Summary & Recommendation

### Model Configuration

The system uses dual language models:
- **OpenAI GPT-4o-mini**: Analysis, calculations, synthesis tasks
- **Gemini 2.5 Pro**: Information retrieval, fact-gathering tasks

Model selection is handled by:
- `get_openai_lm()`: Returns configured OpenAI LM
- `get_gemini_lm()`: Returns configured Gemini LM

## Required Environment Variables

```bash
OPENAI_API_KEY=your_openai_key
GEMINI_API_KEY=your_gemini_key
```

## Dependencies

Install required packages:
```bash
pip install dspy openai google-generativeai python-dotenv
```

## Running the Code

**Generate financial analysis for any stock:**
```bash
# Activate virtual environment
source .venv/bin/activate

# Run analysis for any ticker symbol
python financial_analysis_chain.py NVDA
python financial_analysis_chain.py TSLA
python financial_analysis_chain.py AAPL
python financial_analysis_chain.py META
```

**Show help:**
```bash
python financial_analysis_chain.py --help
```

**Run the simple demo:**
```bash
python dspy_prompt_chain_demo.py
```

## Usage Examples

```bash
# Generate Tesla analysis
python financial_analysis_chain.py TSLA
# Output: ✅ Analysis complete! Report saved to: reports/TSLA_20250624_185032_analysis.md

# Generate Apple analysis  
python financial_analysis_chain.py AAPL
# Output: ✅ Analysis complete! Report saved to: reports/AAPL_20250624_185146_analysis.md

# Generate Meta analysis
python financial_analysis_chain.py META
# Output: ✅ Analysis complete! Report saved to: reports/META_20250624_185642_analysis.md
```

## Supported Tickers

The system includes built-in company mappings for:
- **NVDA**: NVIDIA Corporation (Semiconductors)
- **AAPL**: Apple Inc. (Technology)
- **MSFT**: Microsoft Corporation (Technology)
- **GOOGL**: Alphabet Inc. (Technology)
- **TSLA**: Tesla Inc. (Automotive)
- **META**: Meta Platforms Inc. (Technology)
- **AMZN**: Amazon.com Inc. (E-commerce)

For other tickers, the system will use `[TICKER] Corporation` and `Unknown` sector as defaults.

## Report Output

Each analysis generates a comprehensive markdown report in the `reports/` directory with:
- **Executive Summary**: Investment recommendation and key insights
- **Company Overview**: Business description and market position
- **Industry Analysis**: Sector overview and trends
- **Financial Analysis**: Summary, ratios, and trend analysis
- **Strategic Analysis**: SWOT analysis and management evaluation
- **Investment Analysis**: Valuation, risks, and investment thesis
- **Additional Factors**: Analyst opinions and ESG considerations

Report filename format: `{TICKER}_{YYYYMMDD_HHMMSS}_analysis.md`

## Key Implementation Details

- DSPy signatures use `->` notation for input/output mapping
- Results are accessed via `getattr(result, "field_name", result)` pattern
- Context dictionary maintains state across all 14 steps
- Error handling includes fallback for missing DSPy installation
- The `forward()` method orchestrates the full analysis pipeline

## Troubleshooting Common Issues

- **AttributeError on result access**: Use `getattr(result, "field", result)` pattern
- **API key errors**: Ensure both OPENAI_API_KEY and GEMINI_API_KEY are set
- **DSPy import errors**: Install with `pip install dspy`
- **Model configuration issues**: Check LM initialization in `get_*_lm()` functions