"""固定题集与文档级检索指标；答案字符串检查只作人工审阅辅助。"""
import math
import re
import statistics
from collections import Counter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EvalDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)


class EvalQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1)
    category: Literal["fact", "semantic", "multi_source", "unanswerable"]
    query: str = Field(min_length=1)
    relevant_source_ids: list[str]
    reference_answer: str = Field(min_length=1)
    required_terms: list[list[str]]

    @model_validator(mode="after")
    def validate_labels(self):
        relevant = self.relevant_source_ids
        if len(set(relevant)) != len(relevant):
            raise ValueError("相关文档不能重复")
        if (self.category == "unanswerable") != (len(relevant) == 0):
            raise ValueError("只有无答案题可以没有相关文档")
        if self.category == "multi_source" and len(relevant) < 2:
            raise ValueError("多来源题至少标注两个来源")
        if any(not group or any(not term.strip() for term in group) for group in self.required_terms):
            raise ValueError("答案要点组与候选字符串不能为空")
        return self


class FixedDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: str
    description: str
    annotation: str
    documents: list[EvalDocument] = Field(min_length=1)
    questions: list[EvalQuestion] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_ids(self):
        docs = [doc.id for doc in self.documents]
        questions = [question.id for question in self.questions]
        if len(set(docs)) != len(docs) or len(set(questions)) != len(questions):
            raise ValueError("文档 ID 与题目 ID 各自必须唯一")
        for question in self.questions:
            if not set(question.relevant_source_ids) <= set(docs):
                raise ValueError(f"{question.id} 标注了不存在的文档")
        return self


def retrieval_metrics(ranked_ids: list[str], relevant_ids: list[str]) -> dict | None:
    """先按文档去重；无答案题不定义 IR 指标，不能作为满分或零分计入均值。"""
    relevant = set(relevant_ids)
    if not relevant:
        return None
    ranked = list(dict.fromkeys(ranked_ids))
    metrics = {}
    for k in (1, 3, 5, 8):
        hits = len(set(ranked[:k]) & relevant)
        metrics.update({f"recall@{k}": hits / len(relevant), f"hit@{k}": float(hits > 0),
                        f"all_sources_hit@{k}": float(hits == len(relevant))})
    metrics["mrr@8"] = next((1 / rank for rank, doc in enumerate(ranked[:8], 1) if doc in relevant), 0.0)
    dcg = sum(1 / math.log2(rank + 1) for rank, doc in enumerate(ranked[:5], 1) if doc in relevant)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(5, len(relevant)) + 1))
    metrics["ndcg@5"] = dcg / ideal
    return metrics


def answer_checks(answer: str, question: EvalQuestion) -> dict:
    """字符串覆盖与拒答措辞均非事实正确性、引用支持度或忠实度评分。"""
    normalized = "".join(answer.split()).casefold()
    groups = question.required_terms

    def contains(term):
        # iloc does not cover loc; preserve English word spacing while allowing
        # identifiers next to Chinese text, which has no word separators.
        if re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", term):
            return re.search(r"(?<![A-Za-z_0-9])" + re.escape(term) + r"(?![A-Za-z_0-9])", answer, re.I) is not None
        value = "".join(term.split()).casefold()
        if value.isascii() and value.isdigit():
            return re.search(r"(?<![0-9])" + value + r"(?![0-9])", normalized) is not None
        return value in normalized

    matched = [any(contains(term) for term in group) for group in groups]
    refusal = any(term in normalized for term in ("未提供", "没有提供", "未记录", "没有记录", "证据不足", "无法确定", "无法回答", "未提及", "未包含", "资料不足"))
    return {"keypoint_coverage_heuristic": sum(matched) / len(groups) if groups else None,
            "matched_groups": matched, "refusal_phrase_heuristic": refusal}


def latency_summary(values: list[float]) -> dict | None:
    if not values:
        return None
    ordered = sorted(values)
    return {"p50": round(statistics.median(values), 2),
            "p95": round(ordered[math.ceil(len(values) * .95) - 1], 2),
            "mean": round(statistics.mean(values), 2)}


def aggregate_rows(rows: list[dict], variant: str) -> dict:
    # 运行失败的可答题计为零检索分，避免只对成功请求报告乐观结果。
    scored = [row[variant]["metrics"] for row in rows if row.get(variant, {}).get("metrics") is not None]
    metrics = {key: round(statistics.mean(row[key] for row in scored), 4) for key in scored[0]} if scored else {}
    answers = [row[variant]["answer"] for row in rows if row.get(variant, {}).get("answer", {}).get("status") == "completed"]
    coverage = [answer["checks"]["keypoint_coverage_heuristic"] for answer in answers
                if answer["checks"]["keypoint_coverage_heuristic"] is not None]
    missing = [row[variant]["answer"]["checks"]["refusal_phrase_heuristic"] for row in rows
               if row["category"] == "unanswerable" and row.get(variant, {}).get("answer", {}).get("status") == "completed"]
    return {"questions": len(rows), "ir_questions": len(scored), "metrics": metrics,
            "retrieval_failures": sum(row.get("status") == "error" for row in rows),
            "rerank_status_counts": dict(Counter(row.get("rerank_status", "unknown") for row in rows)) if variant == "reranked" else {},
            "retrieval_ms": latency_summary([row["retrieval_ms"] for row in rows if row.get("status") == "completed" and "retrieval_ms" in row]),
            "rerank_ms": latency_summary([row["rerank_ms"] for row in rows if "rerank_ms" in row]) if variant == "reranked" else None,
            "answers_completed": len(answers),
            "answer_failures": sum(row.get(variant, {}).get("answer", {}).get("status") == "error" for row in rows),
            "answer_ms": latency_summary([answer["elapsed_ms"] for answer in answers]),
            "keypoint_coverage_heuristic": round(statistics.mean(coverage), 4) if coverage else None,
            "unanswerable_answers_completed": len(missing),
            "refusal_phrase_rate_heuristic": round(statistics.mean(missing), 4) if missing else None}
