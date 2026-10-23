"""reorder_service.py — ReorderService 云端 rerank 逻辑测试（不发真实网络请求）。"""
import pytest

from app.rag.reorder_service import ReorderService


class FakeAsyncClient:
    """假 httpx.AsyncClient：post 返回预设响应或抛异常。"""

    def __init__(self, payload=None, exc: Exception | None = None, requests: list | None = None):
        self.payload = payload
        self.exc = exc
        self.requests = requests if requests is not None else []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, **kwargs):
        self.requests.append({"url": url, **kwargs})
        if self.exc is not None:
            raise self.exc

        class _Resp:
            def __init__(self, payload):
                self._payload = payload

            def raise_for_status(self):
                pass

            def json(self):
                return self._payload

        return _Resp(self.payload)


def _make_service(payload=None, exc: Exception | None = None, provider="compatible") -> tuple[ReorderService, FakeAsyncClient]:
    """构造走假 HTTP 客户端的服务实例，返回 (service, client)。"""
    client = FakeAsyncClient(payload=payload, exc=exc)
    service = ReorderService(http_client_factory=lambda **kw: client)
    service.provider, service.enabled = provider, True
    service.api_base_url, service.api_key, service.model = "https://rerank.test/v1", "test", "test-model"
    return service, client


def _rerank_payload(scores: list[float]) -> dict:
    return {"results": [{"index": i, "relevance_score": s} for i, s in enumerate(scores)]}


class TestReorderDocuments:
    async def test_success_returns_sorted_documents(self):
        service, client = _make_service(payload=_rerank_payload([0.3, 0.9, 0.5]))
        result = await service.reorder_documents("查询", ["docA", "docB", "docC"])
        assert result["success"] is True
        assert result["error"] == ""
        docs = result["documents"]
        # 按相似度降序排列
        assert [d["document"] for d in docs] == ["docB", "docC", "docA"]
        assert [d["similarity"] for d in docs] == [0.9, 0.5, 0.3]
        for d in docs:
            assert set(d.keys()) == {"index", "document", "similarity"}
        assert [d["index"] for d in docs] == [1, 2, 0]

    async def test_request_carries_model_query_documents(self):
        service, client = _make_service(payload=_rerank_payload([0.5]))
        service.api_base_url = "https://api.example.com/v1"
        service.api_key = "sk-test"
        service.model = "test-reranker"
        await service.reorder_documents("q", ["a"])
        req = client.requests[0]
        assert req["url"] == "https://api.example.com/v1/rerank"
        assert req["headers"]["Authorization"] == "Bearer sk-test"
        assert req["json"] == {"model": "test-reranker", "query": "q", "documents": ["a"]}

    async def test_success_with_thinking_callback(self):
        calls = []

        async def cb(data):
            calls.append(data)

        service, _ = _make_service(payload=_rerank_payload([0.8, 0.2]))
        result = await service.reorder_documents("q", ["a", "b"], thinking_callback=cb)
        assert result["success"] is True
        # 两次 thinking：开始计算 + 完成详情
        assert len(calls) == 2
        assert calls[0]["stage"] == "reorder"
        assert "2" in calls[0]["content"]
        assert calls[1]["details"]["scores"][0]["score"] == 0.8

    async def test_empty_documents_short_circuits(self):
        service, client = _make_service(payload=_rerank_payload([]))
        result = await service.reorder_documents("q", [])
        assert result == {"success": True, "documents": [], "error": ""}
        assert client.requests == []  # 空列表不发起请求

    async def test_api_failure_returns_error_shape(self):
        service, _ = _make_service(exc=RuntimeError("rerank 服务不可用"))
        result = await service.reorder_documents("q", ["a", "b"])
        assert result["success"] is False
        assert result["documents"] == []
        assert result["status"] == "error"
        assert "保留原排序" in result["error"]

    async def test_no_callback_when_none(self):
        service, _ = _make_service(payload=_rerank_payload([1.0]))
        result = await service.reorder_documents("q", ["a"])
        assert result["success"] is True

    @pytest.mark.parametrize("provider", ["compatible", "dashscope"])
    @pytest.mark.parametrize("payload", [
        {"results": [{"index": 0, "relevance_score": 0.1}]},
        {"results": [{"index": 0, "relevance_score": 0.1}, {"index": 0, "relevance_score": 0.9}]},
        {"results": [{"index": -1, "relevance_score": 0.1}, {"index": 1, "relevance_score": 0.9}]},
        {"results": [{"index": True, "relevance_score": 0.1}, {"index": 0, "relevance_score": 0.9}]},
        {"results": [{"index": 0, "relevance_score": float("nan")}, {"index": 1, "relevance_score": 0.9}]},
        {"results": [{"index": 0, "relevance_score": None}, {"index": 1, "relevance_score": 0.9}]},
        {"results": [{"index": 0, "relevance_score": 0.1}, {"index": 2, "relevance_score": 0.9}]},
        {"results": "invalid"}, [],
    ])
    async def test_invalid_responses_never_return_partial_documents(self, payload, provider):
        service, _ = _make_service(payload={"output": payload} if provider == "dashscope" else payload, provider=provider)
        result = await service.reorder_documents("q", ["a", "b"])
        assert result["success"] is False and result["documents"] == []
        assert result["status"] == "invalid_response"

    async def test_out_of_order_provider_rows_map_to_original_document_indexes(self):
        service, _ = _make_service(payload={"results": [
            {"index": 1, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.2}]})
        result = await service.reorder_documents("q", ["a", "b"])
        assert [row["index"] for row in result["documents"]] == [1, 0]

    async def test_missing_config_and_disabled_never_call_network(self):
        service, client = _make_service()
        service.api_key = ""
        assert (await service.reorder_documents("q", ["a"]))["status"] == "not_configured"
        service.enabled = False
        assert (await service.reorder_documents("q", ["a"]))["status"] == "disabled"
        assert client.requests == []


class TestDashScopeRerank:
    @pytest.mark.parametrize("base_url", ["https://llm-test.cn-beijing.maas.aliyuncs.com/api/v1/", "https://dashscope.aliyuncs.com/api/v1"])
    async def test_native_request_and_out_of_order_response(self, base_url):
        service, client = _make_service(provider="dashscope", payload={"output": {"results": [
            {"index": 1, "relevance_score": .9}, {"index": 0, "relevance_score": .2}]},
            "usage": {"total_tokens": 20}, "request_id": "test-request"})
        service.api_base_url, service.model = base_url.rstrip("/"), "qwen3-rerank"
        result = await service.reorder_documents("问题", ["资料甲", "资料乙"])
        request = client.requests[0]
        assert request["url"] == base_url.rstrip("/") + "/services/rerank/text-rerank/text-rerank"
        assert request["headers"] == {"Authorization": "Bearer test"}
        assert request["json"] == {"model": "qwen3-rerank", "input": {"query": "问题", "documents": ["资料甲", "资料乙"]},
                                   "parameters": {"top_n": 2}}
        assert result["status"] == "applied"
        assert [row["index"] for row in result["documents"]] == [1, 0]
        assert [row["document"] for row in result["documents"]] == ["资料乙", "资料甲"]

    @pytest.mark.parametrize("payload", [{"code": "InvalidApiKey", "message": "test"},
                                         {"results": [{"index": 0, "relevance_score": .9}]},
                                         {"output": None}, {"output": []}, []])
    async def test_native_error_or_missing_output_never_counts_as_success(self, payload):
        service, _ = _make_service(provider="dashscope", payload=payload)
        result = await service.reorder_documents("问题", ["资料甲"])
        assert result["status"] == "invalid_response" and result["documents"] == []

    def test_provider_is_explicit_and_validated(self):
        from pydantic import ValidationError

        from app.core.settings import Settings

        assert Settings(_env_file=None).RERANKER_PROVIDER == "compatible"
        assert Settings(_env_file=None, RERANKER_PROVIDER="dashscope").RERANKER_PROVIDER == "dashscope"
        with pytest.raises(ValidationError):
            Settings(_env_file=None, RERANKER_PROVIDER="unknown")


class TestFormatReorderResult:
    async def test_formats_scores_and_content(self):
        text = await ReorderService.format_reorder_result(
            [{"document": "内容A", "similarity": 0.9123}]
        )
        assert "0.9123" in text
        assert "内容A" in text
