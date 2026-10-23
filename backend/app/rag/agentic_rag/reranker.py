"""按 API 文档索引重排证据，保留来源；失败时原顺序返回，绝不按正文反查来源。"""
import asyncio
import time
from dataclasses import dataclass

from app.core.settings import settings
from app.rag.agentic_rag.evidence import merge_evidence
from app.rag.agentic_rag.schemas import Evidence
from app.rag.reorder_service import ReorderService

CONTEXT_LIMIT = 8  # 融合/引用核对的最大容量；旧版固定题基准仍显式使用 K=8。
MAX_CANDIDATES = 80  # planner 最多两个文本步骤，每步最多 40 个候选


@dataclass
class RankedEvidence:
    evidences: list[Evidence]
    status: str
    candidate_count: int
    elapsed_ms: float

    def summary(self) -> dict:
        return {"status": self.status, "candidate_count": self.candidate_count,
                "selected_count": len(self.evidences), "elapsed_ms": round(self.elapsed_ms, 2)}


class EvidenceReranker:
    def __init__(self, service: ReorderService | None = None):
        self.service = service or ReorderService()

    @property
    def active(self) -> bool:
        return self.service.enabled and self.service.configured

    async def rank(self, query: str, evidences: list[Evidence], limit: int | None = None) -> RankedEvidence:
        limit = settings.RAG_CONTEXT_K if limit is None else limit
        started = time.perf_counter()
        candidates = merge_evidence(evidences, limit=MAX_CANDIDATES)

        def outcome(items, status):
            return RankedEvidence(items[:limit], status, len(candidates), (time.perf_counter() - started) * 1000)

        if not candidates:
            return outcome([], "empty")
        if not self.service.enabled:
            return outcome(candidates, "disabled")
        if not self.service.configured:
            return outcome(candidates, "not_configured")
        if len(candidates) == 1:
            return outcome(candidates, "single_candidate")

        # 以独立的检索评分和重排序评分记录，不能把 RRF 分数当作相关概率。
        try:
            result = await asyncio.wait_for(self.service.reorder_documents(
                query, [f"{item.title}\n{item.content}" for item in candidates]), timeout=self.service.timeout)
            if not result["success"]:
                return outcome(candidates, result.get("status", "error"))
            ranked = []
            indexes = [item["index"] for item in result["documents"]]
            if (len(indexes) != len(candidates) or any(type(index) is not int for index in indexes)
                    or set(indexes) != set(range(len(candidates)))):
                return outcome(candidates, "invalid_response")
            for row in result["documents"]:
                evidence = candidates[row["index"]]
                ranked.append(evidence.model_copy(update={"metadata": {
                    **evidence.metadata, "retrieval_score": evidence.score,
                    "rerank_score": row["similarity"], "rerank_model": self.service.model}}))
            return outcome(ranked, "applied")
        except TimeoutError:
            return outcome(candidates, "timeout")
        except Exception:
            return outcome(candidates, "error")
