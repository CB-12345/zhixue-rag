"""云端重排序服务：兼容 rerank 协议与百炼 DashScope 原生协议。

接口约定：POST {RERANKER_API_BASE_URL}/rerank，payload {"model", "query", "documents"}，
响应 {"results": [{"index": int, "relevance_score": float}, ...]}。
DashScope 使用 /services/rerank/text-rerank/text-rerank、input/parameters 和 output.results。
API 未配置或调用失败时返回 success=False，由调用方按原顺序 + similarity 0 降级。
"""

import math
from typing import Any

import httpx

from app.core.logger_handler import logger
from app.core.settings import settings


class ReorderService:
    """文档重排序服务（云端 rerank API）"""

    def __init__(self, http_client_factory=None):
        self.provider = settings.RERANKER_PROVIDER
        self.api_base_url = (settings.RERANKER_API_BASE_URL or "").rstrip("/")
        self.api_key = settings.RERANKER_API_KEY
        self.model = settings.RERANKER_MODEL
        self.enabled = settings.RERANKER_ENABLED
        self.timeout = settings.RERANKER_TIMEOUT_SECONDS
        self.http_client_factory = http_client_factory or httpx.AsyncClient

    @property
    def configured(self) -> bool:
        return bool(self.api_base_url and self.api_key and self.model)

    async def _rerank(self, query: str, documents: list[str]) -> list[float]:
        """调用 rerank API，按文档原顺序返回相关性分数。"""
        if self.provider == "dashscope":
            path = "/services/rerank/text-rerank/text-rerank"
            payload = {"model": self.model, "input": {"query": query, "documents": documents},
                       "parameters": {"top_n": len(documents)}}
        else:
            path = "/rerank"
            payload = {"model": self.model, "query": query, "documents": documents}
        async with self.http_client_factory(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.api_base_url}{path}",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
        if self.provider == "dashscope":
            data = data.get("output") if isinstance(data, dict) else None
        results = data.get("results") if isinstance(data, dict) else None
        if not isinstance(results, list) or len(results) != len(documents):
            raise ValueError("重排序响应缺少完整的文档评分")
        scores = [None] * len(documents)
        for result in results:
            if not isinstance(result, dict):
                raise ValueError("重排序响应格式错误")
            index, score = result.get("index"), result.get("relevance_score")
            if (type(index) is not int or not 0 <= index < len(documents)
                    or scores[index] is not None or isinstance(score, bool)):
                raise ValueError("重排序响应的文档索引或评分无效")
            try:
                score = float(score)
            except (TypeError, ValueError) as exc:
                raise ValueError("重排序评分不是有效数值") from exc
            if not math.isfinite(score):
                raise ValueError("重排序评分必须为有限数值")
            scores[index] = score
        return scores

    async def reorder_documents(self, query: str, documents: list[str], thinking_callback=None) -> dict[str, Any]:
        """
        对文档进行重排序
        :param query: 查询语句
        :param documents: 文档列表
        :param thinking_callback: 思考过程回调函数
        :return: 包含重排序结果的字典，格式为：
                 {"success": bool, "documents": List[Dict], "error": str}
        """
        try:
            if not documents:
                return {"success": True, "documents": [], "error": ""}
            if not self.enabled:
                return {"success": False, "documents": [], "error": "重排序已关闭", "status": "disabled"}
            if not self.configured:
                return {"success": False, "documents": [], "error": "重排序服务未配置", "status": "not_configured"}

            if thinking_callback:
                await thinking_callback({"type": "thinking", "stage": "reorder", "content": f"正在计算 {len(documents)} 个文档的相关性分数..."})

            scores = await self._rerank(query, documents)

            # 构建结果列表
            scored_documents = []
            for index, (doc, score) in enumerate(zip(documents, scores)):
                scored_documents.append({"index": index, "document": doc, "similarity": score})

            if thinking_callback:
                score_details = []
                for i, (doc, score) in enumerate(zip(documents, scores), 1):
                    score_details.append({"index": i, "score": round(score, 4), "preview": doc[:100] + "..." if len(doc) > 100 else doc})
                await thinking_callback(
                    {"type": "thinking", "stage": "reorder", "content": f"已计算完成 {len(documents)} 个文档的相关性分数，按分数降序排序", "details": {"scores": score_details}}
                )

            # 按相似度分数降序排序
            sorted_docs = sorted(scored_documents, key=lambda x: x["similarity"], reverse=True)
            logger.info(f"【重排序服务】文档重排序成功，返回 {len(sorted_docs)} 个文档")

            return {"success": True, "documents": sorted_docs, "error": "", "status": "applied"}
        except Exception as e:
            status = "timeout" if isinstance(e, httpx.TimeoutException) else "invalid_response" if isinstance(e, ValueError) else "error"
            logger.warning(f"【重排序服务】重排序未应用: {status} ({type(e).__name__})")
            return {"success": False, "documents": [], "error": "重排序服务调用失败，保留原排序", "status": status}

    @staticmethod
    async def format_reorder_result(sorted_docs: list[dict]) -> str:
        """
        格式化重排序结果
        :param sorted_docs: 重排序后的文档列表
        :return: 格式化后的字符串
        """
        formatted_result = "重排序后的文档列表：\n"
        for i, doc in enumerate(sorted_docs, 1):
            formatted_result += f"{i}. 相似度: {doc.get('similarity', 0):.4f}\n"
            formatted_result += f"   内容: {doc.get('document', '')}\n\n"
        return formatted_result


# 全局重排序服务实例
reorder_service = ReorderService()
