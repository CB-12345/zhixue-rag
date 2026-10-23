import asyncio
import inspect
import time
from collections.abc import Callable
from typing import Any

from app.core.logger_handler import logger
from app.core.settings import settings
from app.rag.agentic_rag.conversation import ConversationQueryResolver
from app.rag.agentic_rag.evaluator import AnswerabilityEvaluator, is_local_only_query
from app.rag.agentic_rag.evidence import merge_evidence, prepare_evidence_context
from app.rag.agentic_rag.local_retriever import LocalRetriever
from app.rag.agentic_rag.planner import AgenticRagPlanner
from app.rag.agentic_rag.reranker import EvidenceReranker
from app.rag.agentic_rag.schemas import AgenticRagResult, AnswerabilityResult, Evidence, RetrievalPlan, RetrievalStep
from app.rag.agentic_rag.web_search import WebSearchClient

ThinkingCallback = Callable[[dict[str, Any]], Any]
INSUFFICIENT_EVIDENCE_MESSAGE = "现有资料中没有足够证据回答这个问题，无法确定所问的信息。请补充相关资料，或明确需要核对的内容。"
VERIFICATION_UNAVAILABLE_MESSAGE = "暂时无法完成资料核对，因此无法依据现有资料给出可靠回答。请稍后重试。"


class AgenticRagService:
    def __init__(
        self,
        planner: AgenticRagPlanner | None = None,
        local_retriever: LocalRetriever | None = None,
        evaluator: AnswerabilityEvaluator | None = None,
        web_search_client: WebSearchClient | None = None,
        reranker: EvidenceReranker | None = None,
        query_resolver: ConversationQueryResolver | None = None,
    ):
        self.planner = planner or AgenticRagPlanner()
        self.local_retriever = local_retriever or LocalRetriever()
        self.evaluator = evaluator or AnswerabilityEvaluator()
        self.web_search_client = web_search_client or WebSearchClient()
        self.reranker = reranker or EvidenceReranker()
        self.query_resolver = query_resolver or ConversationQueryResolver()

    async def run(
        self,
        query: str,
        user_id: str,
        thinking_callback: ThinkingCallback | None = None,
        history: list[tuple[str, str]] | None = None,
    ) -> AgenticRagResult:
        t_total = time.perf_counter()
        resolution = await self.query_resolver.resolve(query, history)
        query_resolution = {"status": resolution.status, "original_query": query,
                            "retrieval_query": resolution.query, "method": resolution.method}
        if resolution.status != "unchanged":
            await self._emit(thinking_callback, "query_rewrite",
                             resolution.clarification or "已结合最近对话明确提问对象。",
                             query_resolution)
        if resolution.clarification:
            return AgenticRagResult(
                context="", evidences=[],
                plan=RetrievalPlan(need_retrieval=False, steps=[], reason="Follow-up needs clarification."),
                refusal_message=resolution.clarification, query_resolution=query_resolution)
        # 保留当前原话中的范围、否定和所问属性，避免改写遗漏要求后被误判可答。
        effective_query = (f"本轮原始问题：{query}\n补全指代后的问题：{resolution.query}"
                           if resolution.status == "resolved" else query)
        t0 = time.perf_counter()
        plan = await self.planner.plan(resolution.query)
        if resolution.status == "resolved" and not plan.need_retrieval:
            plan = RetrievalPlan(need_retrieval=True, steps=[RetrievalStep(
                tool="hybrid_search", query=resolution.query, top_k=5)], reason="Retrieve evidence for resolved follow-up.")
        planner_ms = (time.perf_counter() - t0) * 1000
        logger.info(f"【RAG耗时】planner={planner_ms:.0f}ms source={plan.metadata.get('source', 'unknown')} need_retrieval={plan.need_retrieval} steps={len(plan.steps)}")
        await self._emit(
            thinking_callback,
            "agentic_plan",
            "Planned retrieval strategy.",
            {
                "query": resolution.query,
                "source": plan.metadata.get("source", "unknown"),
                "steps": [step.model_dump() for step in plan.steps],
                "need_retrieval": plan.need_retrieval,
                "allow_web_fallback": plan.allow_web_fallback,
                "step_count": len(plan.steps),
                "reason": plan.reason,
            },
        )

        if not plan.need_retrieval:
            total_ms = (time.perf_counter() - t_total) * 1000
            logger.info(f"【RAG耗时】total={total_ms:.0f}ms (无需检索短路) planner={planner_ms:.0f}ms")
            return AgenticRagResult(context="", evidences=[], plan=plan, answerability=None, used_web=False,
                                    query_resolution=query_resolution)

        graph_steps = [step for step in plan.steps if step.tool == "search_graph"]
        text_steps = [step for step in plan.steps if step.tool != "search_graph"]
        if self.reranker.active:
            # 保留 planner 的原始计划；执行时扩大召回，之后才截取上下文。
            text_steps = [step.model_copy(update={"top_k": max(step.top_k, settings.RERANKER_CANDIDATE_K)})
                          for step in text_steps]

        await self._emit(
            thinking_callback,
            "local_retrieval",
            "Searching local sources.",
            {"status": "searching"},
        )
        # 图检索（含 LLM 实体抽取）与文本检索并行，避免 LLM 抽取拉长整体延迟
        t_retrieval = time.perf_counter()
        graph_task = asyncio.create_task(self._search_graph_only(user_id, graph_steps))
        text_task = asyncio.create_task(self.local_retriever.search(user_id, text_steps))
        local_evidences = []
        graph_evidences = []
        try:
            if graph_steps:
                # 图检索含额外 LLM 抽取；超时/失败不阻塞已完成的文本检索。
                try:
                    graph_evidences = await graph_task
                except Exception as e:
                    logger.warning(f"图谱检索失败: {type(e).__name__}")
            if text_steps:
                local_evidences = await text_task
        finally:
            # 外部取消或任一分支失败时，退出前回收所有子任务。
            for task in (graph_task, text_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(graph_task, text_task, return_exceptions=True)
        retrieval_ms = (time.perf_counter() - t_retrieval) * 1000
        logger.info(f"【RAG耗时】local_retrieval={retrieval_ms:.0f}ms graph_steps={len(graph_steps)} text_steps={len(text_steps)} evidence={len(local_evidences)}")
        local_evidences = [*local_evidences, *graph_evidences]
        ranked = await self.reranker.rank(effective_query, local_evidences)
        local_evidences = ranked.evidences
        logger.info(f"【RAG重排序】{ranked.summary()}")
        if self.reranker.active:
            await self._emit(thinking_callback, "reorder",
                             "已按问题相关性整理参考资料。" if ranked.status == "applied" else "已保留检索结果，继续整理参考资料。",
                             ranked.summary())

        await self._emit(
            thinking_callback,
            "local_retrieval",
            "Retrieved local evidence.",
            {
                "status": "evidence",
                "queries": [step.model_dump() for step in plan.steps],
                "evidence_count": len(local_evidences),
                "results": [self._evidence_preview(evidence) for evidence in local_evidences],
            },
        )
        await asyncio.sleep(0)  # 分帧：让各阶段落入不同事件循环 tick，避免一次性涌入前端

        # 判定、模型上下文与最终来源必须使用同一份实际可容纳的正文。
        local_evidences = prepare_evidence_context(local_evidences).evidences
        t_eval = time.perf_counter()
        answerability = await self.evaluator.evaluate(effective_query, local_evidences)
        eval_ms = (time.perf_counter() - t_eval) * 1000
        logger.info(f"【RAG耗时】evaluator={eval_ms:.0f}ms answerable={answerability.answerable} web_queries={answerability.web_queries}")
        await self._emit(
            thinking_callback,
            "answerability",
            "Evaluated local answerability.",
            {
                "answerable": answerability.answerable,
                "confidence": answerability.confidence,
                "reason": answerability.reason,
                "web_queries": answerability.web_queries,
            },
        )
        await asyncio.sleep(0)

        t_web = time.perf_counter()
        web_evidences = await self._maybe_search_web(
            effective_query, answerability, thinking_callback, local_only=resolution.local_only)
        web_ms = (time.perf_counter() - t_web) * 1000
        # 需要外部补证时保留其位置，避免扩大召回后本地资料占满全部上下文。
        # 本地默认 K=3；必要的网页仍可补充，最终融合继续遵守既有 8 条上限。
        local_budget = min(settings.RAG_CONTEXT_K, 8 - min(3, len(web_evidences)))
        fused_evidences = merge_evidence([*local_evidences[:local_budget], *web_evidences])
        prepared = prepare_evidence_context(fused_evidences)
        fused_evidences = prepared.evidences
        if web_evidences:
            # 搜到网页不等于补齐答案；必须重新核对实际送入上下文的最终资料。
            answerability = await self.evaluator.evaluate(effective_query, fused_evidences)
            await self._emit(thinking_callback, "answerability", "已核对补充资料能否支持回答。",
                             {"answerable": answerability.answerable, "phase": "final", "reason": answerability.reason})
        refusal_message = ""
        if not answerability.answerable or not fused_evidences:
            refusal_message = (VERIFICATION_UNAVAILABLE_MESSAGE if answerability.status == "unavailable"
                               else INSUFFICIENT_EVIDENCE_MESSAGE)
            fused_evidences = []
        context = "" if refusal_message else prepared.text
        logger.info(f"【RAG证据门控】allowed={not bool(refusal_message)} status={answerability.status} evidence={len(fused_evidences)}")
        total_ms = (time.perf_counter() - t_total) * 1000
        logger.info(f"【RAG耗时】total={total_ms:.0f}ms planner={planner_ms:.0f}ms "
                    f"retrieval={retrieval_ms:.0f}ms evaluator={eval_ms:.0f}ms web={web_ms:.0f}ms")
        await self._emit(
            thinking_callback,
            "evidence_fusion",
            "Merged local and web evidence.",
            {
                "local_count": len(local_evidences),
                "web_count": len(web_evidences),
                "fused_count": len(fused_evidences),
            },
        )
        await asyncio.sleep(0)

        await self._emit(
            thinking_callback,
            "context_ready",
            "Prepared evidence context.",
            {"context_chars": len(context), "evidence_count": len(fused_evidences)},
        )

        return AgenticRagResult(
            context=context,
            evidences=fused_evidences,
            plan=plan,
            answerability=answerability,
            used_web=bool(web_evidences),
            rerank=ranked.summary(),
            refusal_message=refusal_message,
            query_resolution=query_resolution,
        )

    async def _maybe_search_web(
        self,
        query: str,
        answerability: AnswerabilityResult,
        thinking_callback: ThinkingCallback | None,
        local_only: bool = False,
    ) -> list[Evidence]:
        """web 兜底：answerability 判不可答（或显式给出 web_queries）即触发。

        planner 的 allow_web_fallback 仅作规划提示、不再一票否决——
        本地知识缺口（检索回来的证据与问题不相关）同样需要搜网；
        硬开关收敛到 WebSearchClient 自身（WEB_SEARCH_ENABLED/provider/key）。
        """
        if local_only or is_local_only_query(query) or answerability.status == "unavailable":
            return []
        if answerability.answerable and not answerability.web_queries:
            return []

        queries = answerability.web_queries or [query]
        web_evidences: list[Evidence] = []
        await self._emit(
            thinking_callback,
            "web_search",
            "Searching external sources.",
            {"status": "searching"},
        )
        for web_query in queries:
            web_evidences.extend(await self.web_search_client.search(web_query, max_results=5))

        await self._emit(
            thinking_callback,
            "web_search",
            "Searched web fallback evidence.",
            {
                "status": "evidence",
                "queries": queries,
                "evidence_count": len(web_evidences),
                "results": [self._evidence_preview(evidence) for evidence in web_evidences],
            },
        )
        return web_evidences

    async def _search_graph_only(self, user_id: str, steps: list) -> list[Evidence]:
        """仅执行 search_graph 步骤（无其他文本步骤时也不抛错）。"""
        if not steps:
            return []
        try:
            return await asyncio.wait_for(self.local_retriever.search(user_id, steps),
                                          timeout=settings.GRAPH_RETRIEVAL_TIMEOUT_SECONDS)
        except TimeoutError:
            logger.warning("图谱补充检索超时，已取消该分支，继续核对文本证据")
            return []

    async def _emit(
        self,
        thinking_callback: ThinkingCallback | None,
        stage: str,
        content: str,
        details: dict[str, Any],
    ) -> None:
        if thinking_callback is None:
            return

        event = {"type": "thinking", "stage": stage, "content": content, "details": details}
        result = thinking_callback(event)
        if inspect.isawaitable(result):
            await result

    @staticmethod
    def _evidence_preview(evidence: Evidence) -> dict[str, Any]:
        return {
            "id": evidence.id,
            "source": evidence.source,
            "title": evidence.title,
            "score": evidence.score,
            "url": evidence.url,
            "preview": evidence.content[:500],
        }
