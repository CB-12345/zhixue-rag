"""background_init 测试：全局模型预热必须 fail-soft。

- .env 无全局 key 是合法状态（key 按用户存 DB），启动预热缺 key 时只告警、
  各模型实例置 None，且 models_ready 必须置位（否则 _init_note_service 永久挂起）。
"""
import pytest


async def _clear_all_model_env(monkeypatch):
    from app.core.settings import settings

    for name in (
        "OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL_NAME",
        "EMBED_BASE_URL", "EMBED_API_KEY", "EMBED_MODEL_NAME",
        "VISION_BASE_URL", "VISION_API_KEY", "VISION_MODEL_NAME",
    ):
        monkeypatch.setattr(settings, name, "", raising=False)


async def test_init_models_failsoft_when_no_global_key(monkeypatch):
    """无全局 key 时预热不崩溃：模型为 None、models_ready 照常置位。"""
    await _clear_all_model_env(monkeypatch)

    from app.core.background_init import _BackgroundInitManager

    mgr = _BackgroundInitManager()
    await mgr._init_models()

    assert mgr.chat_model is None
    assert mgr.embed_model is None
    assert mgr.vision_model is None
    assert mgr.models_ready.is_set()


async def test_init_models_sets_ready_even_when_embed_missing(monkeypatch):
    """对话可用但嵌入部分配置(缺 key)时：对话预热成功、嵌入告警置 None、ready 照常置位。"""
    await _clear_all_model_env(monkeypatch)
    from app.core.settings import settings
    monkeypatch.setattr(settings, "OPENAI_BASE_URL", "http://localhost:11434/v1", raising=False)
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "ollama", raising=False)
    # 仅配 EMBED_BASE_URL 不配 EMBED_API_KEY → 触发 EmbedModelFactory 的"不完整配置"告警
    monkeypatch.setattr(settings, "EMBED_BASE_URL", "http://localhost:11434/v1", raising=False)

    from app.core.background_init import _BackgroundInitManager

    mgr = _BackgroundInitManager()
    await mgr._init_models()

    assert mgr.chat_model is not None
    assert mgr.embed_model is None
    assert mgr.models_ready.is_set()


@pytest.mark.parametrize("failing_model", ["chat", "vision"])
async def test_optional_model_init_failure_does_not_prevent_index_worker(monkeypatch, failing_model):
    from app.core.background_init import _BackgroundInitManager
    from app.utils import factory

    embed = object()

    def fail(self):
        raise RuntimeError("optional model unavailable")

    monkeypatch.setattr(factory, "resolve_chat_config", lambda: {"api_key": "test"})
    monkeypatch.setattr(factory, "create_planner_chat_openai", lambda: None)
    monkeypatch.setattr(factory.ChatModelFactory, "generator", fail if failing_model == "chat" else lambda self: object())
    monkeypatch.setattr(factory.VisionModelFactory, "generator", fail if failing_model == "vision" else lambda self: object())
    monkeypatch.setattr(factory.EmbedModelFactory, "generator", lambda self: embed)
    manager = _BackgroundInitManager()
    steps = []

    async def skip_external_init():
        pass

    for method in ("_init_graph", "_init_note_service", "_init_vector_store", "_init_reranker"):
        monkeypatch.setattr(manager, method, skip_external_init)
    monkeypatch.setattr(manager, "_start_graph_worker", lambda: steps.append("workers_started"))
    await manager._initialize_all()
    assert steps == ["workers_started"]
    assert manager.embed_model is embed and manager.models_ready.is_set()
    assert getattr(manager, failing_model + "_model") is None
