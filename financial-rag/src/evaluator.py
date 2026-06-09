from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from pipeline import FinancialRAGPipeline, QueryFilters, build_default_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports"
DEFAULT_RESULTS_PATH = DEFAULT_REPORTS_DIR / "ragas_results.json"

LOGGER = logging.getLogger(__name__)

@dataclass(frozen=True)
class EvaluationExample:
    question: str
    ground_truth: str
    company_name: str | None = None
    filing_type: str | None = None

DEFAULT_EVALUATION_SET: tuple[EvaluationExample, ...] = (
    EvaluationExample("What were Apple's main net sales categories?", "Apple reports net sales by products and services, with products including iPhone, Mac, iPad, and Wearables, Home and Accessories.", "Apple", "10-K"),
    EvaluationExample("What does Amazon report as its major business segments?", "Amazon reports segments including North America, International, and AWS.", "Amazon", "10-K"),
    EvaluationExample("What are Alphabet's primary revenue sources?", "Alphabet primarily earns revenue from advertising, subscriptions, platforms, devices, and Google Cloud.", "Alphabet", "10-K"),
    EvaluationExample("What are Berkshire Hathaway's major operating groups?", "Berkshire Hathaway reports diversified operations including insurance, rail, utilities and energy, manufacturing, service, and retailing.", "Berkshire Hathaway", "10-K"),
    EvaluationExample("What does Johnson and Johnson report as major business segments?", "Johnson and Johnson reports business through healthcare-focused segments such as Innovative Medicine and MedTech.", "Johnson and Johnson", "10-K"),
    EvaluationExample("What table shows Apple's net sales by category?", "Apple's filing includes a table presenting net sales by category and reportable segment.", "Apple", "10-K"),
    EvaluationExample("What table shows Amazon AWS net sales?", "Amazon's filing includes segment tables that show AWS net sales and operating income.", "Amazon", "10-K"),
    EvaluationExample("What table shows Alphabet revenues by type?", "Alphabet's filing includes revenue tables by Google advertising, subscriptions, platforms, devices, and cloud.", "Alphabet", "10-K"),
    EvaluationExample("What does Berkshire's quarterly filing say about insurance underwriting?", "Berkshire's 10-Q discusses insurance underwriting results and investment income as part of insurance operations.", "Berkshire Hathaway", "10-Q"),
    EvaluationExample("What does Johnson and Johnson's quarterly filing say about MedTech?", "Johnson and Johnson's 10-Q discusses MedTech sales and operating performance.", "Johnson and Johnson", "10-Q"),
    EvaluationExample("Which company reports AWS as a segment?", "Amazon reports AWS as one of its operating segments.", None, "10-K"),
    EvaluationExample("Which company reports iPhone net sales?", "Apple reports iPhone net sales.", None, "10-K"),
    EvaluationExample("Which company reports Google Cloud revenue?", "Alphabet reports Google Cloud revenue.", None, "10-K"),
    EvaluationExample("How do filings discuss foreign currency risk?", "The filings discuss foreign currency risk as exchange-rate movements that can affect revenue, costs, assets, liabilities, or cash flows.", None, "10-K"),
    EvaluationExample("How do filings discuss supply chain risk?", "The filings discuss supply chain risk as disruptions or shortages that may affect production, costs, availability, or customer demand.", None, "10-K"),
    EvaluationExample("How do filings discuss regulatory risk?", "The filings discuss regulatory risk as laws, investigations, compliance duties, or policy changes that can affect operations and financial results.", None, "10-K"),
    EvaluationExample("How do filings discuss AI infrastructure investment?", "The filings discuss AI infrastructure investment through data centers, servers, chips, cloud infrastructure, or capital expenditures.", None, "10-Q"),
    EvaluationExample("How do filings discuss liquidity?", "The filings discuss liquidity through cash, cash equivalents, marketable securities, operating cash flow, debt, and capital resources.", None, "10-K"),
)

def ensure_reports_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def build_ragas_records(pipeline: FinancialRAGPipeline, examples: tuple[EvaluationExample, ...]) -> list[dict[str,Any]]:
    records: list[dict[str,Any]] = []
    for example in examples:
        filters = QueryFilters(example.company_name, example.filing_type)
        response = pipeline.answer_question(example.question, filters)
        contexts = [source.text for source in response.sources]
        records.append(build_record(example, response.answer, contexts))
    return records

def build_record(example: EvaluationExample, answer: str, contexts: list[str]) -> dict[Any, str]:
    return {
        "question" : example.question,
        "answer": answer,
        "ground_truth" : example.ground_truth,
        "contexts": contexts,
    }

def run_ragas(records: list[dict[str,Any]]) -> Any:
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_correctness, context_precision, context_recall, faithfulness
    except ImportError as exc:
        raise ImportError("Install ragas and datasets before running evaluation.") from exc
    
    dataset = Dataset.from_list(records)
    metrics = [faithfulness, answer_correctness, context_precision, context_recall]
    return evaluate(dataset,metrics=metrics)

def save_results(result: Any, records: list[dict[str,Any]], path:Path) -> None:
    ensure_reports_dir(path.parent)
    payload = {"scores": result_to_dict(result), "examples": records}
    try:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError as exc:
        raise OSError(f"Could not save RAGAS results to {path}") from exc

def result_to_dict(result: Any) -> dict[str, Any]:
    if hasattr(result, "to_pandas"):
        return result.to_pandas().mean(numeric_only=True).to_dict()
    return dict(result)
    
def save_evaluation_set(path: Path = DEFAULT_REPORTS_DIR / "evaluation_set.json") -> None :
    ensure_reports_dir(path.parent)
    payload = [ asdict(example) for example in DEFAULT_EVALUATION_SET]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

def evaluate_pipeline(
    pipeline: FinancialRAGPipeline | None = None,
    results_path: Path = DEFAULT_RESULTS_PATH,
) -> Any:
    
    active_pipeline = pipeline or build_default_pipeline()
    save_evaluation_set()
    records = build_ragas_records(active_pipeline, DEFAULT_EVALUATION_SET)
    result = run_ragas(records)
    save_results(result, records, results_path)
    return result
