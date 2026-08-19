from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import json
import math
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    lengths = {len(questions), len(answers), len(contexts), len(ground_truths)}
    if len(lengths) != 1:
        raise ValueError(
            "questions, answers, contexts and ground_truths must have equal lengths"
        )
    if not questions:
        return _empty_evaluation()

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        from config import OPENAI_API_KEY, OPENAI_BASE_URL, LLM_MODEL

        eval_kwargs = {}
        if OPENAI_API_KEY:
            try:
                from langchain_openai import ChatOpenAI
                chat_kwargs = {
                    "model": LLM_MODEL,
                    "api_key": OPENAI_API_KEY,
                    "temperature": 0,
                    "timeout": 15.0,
                    "max_retries": 1,
                }
                if OPENAI_BASE_URL:
                    chat_kwargs["base_url"] = OPENAI_BASE_URL
                eval_kwargs["llm"] = ChatOpenAI(**chat_kwargs)
            except Exception as e:
                print(f"  Warning: ChatOpenAI initialization failed: {e}")

            try:
                from langchain_community.embeddings import HuggingFaceEmbeddings
                eval_kwargs["embeddings"] = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            except Exception:
                pass

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            **eval_kwargs,
        )
        dataframe = result.to_pandas()
        metric_names = (
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        )
        per_question = []
        for _, row in dataframe.iterrows():
            per_question.append(EvalResult(
                question=str(row["question"]),
                answer=str(row["answer"]),
                contexts=list(row["contexts"]),
                ground_truth=str(row["ground_truth"]),
                **{name: _finite_float(row.get(name, 0.0)) for name in metric_names},
            ))

        aggregate = {
            name: _mean([getattr(item, name) for item in per_question])
            for name in metric_names
        }
        return {**aggregate, "per_question": per_question}
    except Exception as exc:
        print(f"  RAGAS evaluation unavailable: {exc}")
        per_question = [
            EvalResult(question, answer, list(context), truth, 0.0, 0.0, 0.0, 0.0)
            for question, answer, context, truth
            in zip(questions, answers, contexts, ground_truths)
        ]
        return {**_empty_evaluation(), "per_question": per_question}


def _empty_evaluation() -> dict:
    return {
        "faithfulness": 0.0,
        "answer_relevancy": 0.0,
        "context_precision": 0.0,
        "context_recall": 0.0,
        "per_question": [],
    }


def _finite_float(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    if bottom_n <= 0:
        return []
    diagnostic_tree = {
        "faithfulness": (
            "The answer contains claims unsupported by the retrieved context.",
            "Tighten the grounded-answer prompt and lower generation temperature.",
        ),
        "context_recall": (
            "The retriever missed information required by the reference answer.",
            "Review chunk boundaries and expand hybrid retrieval coverage.",
        ),
        "context_precision": (
            "The retrieved set contains too many irrelevant chunks.",
            "Tune top-k, improve reranking, or add a metadata filter.",
        ),
        "answer_relevancy": (
            "The answer does not directly address the question.",
            "Make the answer prompt more explicit and preserve the query intent.",
        ),
    }
    metric_names = tuple(diagnostic_tree)
    analyzed = []
    for result in eval_results:
        metric_values = {
            name: _finite_float(getattr(result, name, 0.0)) for name in metric_names
        }
        worst_metric = min(metric_values, key=metric_values.get)
        diagnosis, suggested_fix = diagnostic_tree[worst_metric]
        analyzed.append({
            "question": result.question,
            "average_score": _mean(list(metric_values.values())),
            "worst_metric": worst_metric,
            "score": metric_values[worst_metric],
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        })
    return sorted(analyzed, key=lambda item: item["average_score"])[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
