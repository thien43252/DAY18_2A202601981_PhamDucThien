from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
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


def _safe_float(value) -> float:
    """RAGAS trả NaN khi metric không tính được → quy về 0.0."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if value != value else value  # NaN check


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation.

    4 metrics:
      - faithfulness       : answer có bịa so với context không (generation)
      - answer_relevancy   : answer có trả lời đúng câu hỏi không (generation)
      - context_precision  : context lấy về có bao nhiêu phần liên quan (retrieval)
      - context_recall     : ground_truth được context bao phủ bao nhiêu (retrieval)
    """
    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    empty = {m: 0.0 for m in metric_names} | {"per_question": []}

    try:
        from ragas import evaluate
        from ragas.metrics import (faithfulness, answer_relevancy,
                                   context_precision, context_recall)
        from datasets import Dataset

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })
        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy,
                                            context_precision, context_recall])
        df = result.to_pandas()

        per_question = [
            EvalResult(
                question=row["question"],
                answer=row["answer"],
                contexts=list(row["contexts"]),
                ground_truth=row["ground_truth"],
                faithfulness=_safe_float(row.get("faithfulness", 0.0)),
                answer_relevancy=_safe_float(row.get("answer_relevancy", 0.0)),
                context_precision=_safe_float(row.get("context_precision", 0.0)),
                context_recall=_safe_float(row.get("context_recall", 0.0)),
            )
            for _, row in df.iterrows()
        ]

        aggregate = {}
        for m in metric_names:
            values = [getattr(r, m) for r in per_question]
            aggregate[m] = sum(values) / len(values) if values else 0.0

        return {**aggregate, "per_question": per_question}

    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed: {e}")
        return empty


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    # Diagnostic Tree: metric yếu nhất → nguyên nhân gốc → hướng sửa
    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating — answer chứa thông tin không có trong context",
                         "Siết prompt ('chỉ dùng context'), giảm temperature, thêm citation bắt buộc"),
        "context_recall": ("Missing relevant chunks — retrieval bỏ sót thông tin trong ground truth",
                           "Tăng top_k, cải thiện chunking (hierarchical/structure), thêm BM25 vào hybrid"),
        "context_precision": ("Too many irrelevant chunks — context bị pha loãng bởi nhiễu",
                              "Thêm/siết reranking, lọc theo metadata (version, category), giảm top_k sau rerank"),
        "answer_relevancy": ("Answer doesn't match question — trả lời lệch trọng tâm câu hỏi",
                             "Cải thiện prompt template, yêu cầu trả lời trực tiếp câu hỏi, query rewriting"),
    }
    metric_names = list(diagnostic_tree.keys())

    scored = []
    for r in eval_results:
        metrics = {m: getattr(r, m) for m in metric_names}
        avg = sum(metrics.values()) / len(metrics)
        worst_metric = min(metrics, key=metrics.get)
        diagnosis, fix = diagnostic_tree[worst_metric]
        scored.append({
            "question": r.question,
            "answer": r.answer,
            "ground_truth": r.ground_truth,
            "contexts": r.contexts,
            "avg_score": round(avg, 4),
            "worst_metric": worst_metric,
            "score": round(metrics[worst_metric], 4),
            "metrics": {m: round(v, 4) for m, v in metrics.items()},
            "diagnosis": diagnosis,
            "suggested_fix": fix,
        })

    scored.sort(key=lambda x: x["avg_score"])
    return scored[:bottom_n]


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
