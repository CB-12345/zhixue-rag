import asyncio
from datetime import datetime, timezone

from fastapi import HTTPException
from sqlalchemy import func, select

from app.core.logger_handler import logger
from app.db.db_config import AsyncSessionLocal
from app.models.chat_history import ChatMessage, ChatSession


def _utc_isoformat(value: datetime | None) -> str | None:
    """MySQL DATETIME 不保留时区；会话时间按数据库的 UTC 约定传给浏览器。"""
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


class DatabaseSessionManager:
    """基于数据库的会话管理器"""

    def __init__(self):
        """初始化会话管理器"""
        self._lock = asyncio.Lock()

    @classmethod
    async def create(cls) -> "DatabaseSessionManager":
        """
        异步创建并初始化 DatabaseSessionManager
        :return: 初始化完成的 DatabaseSessionManager 实例
        """
        instance = cls()
        logger.info("【数据库会话管理】初始化完成")
        return instance

    async def get_session(self, session_id: str, user_id: str, *, include_sources: bool = False) -> dict:
        """获取会话"""
        async with AsyncSessionLocal() as db:
            # 尝试查找会话，验证属于该用户
            result = await db.run_sync(
                lambda session: session.query(ChatSession).filter(ChatSession.id == session_id, ChatSession.user_id == user_id).first()
            )

            if result:
                # 获取会话历史
                messages = await db.run_sync(
                    lambda session: session.query(ChatMessage).filter(ChatMessage.session_id == result.id).order_by(ChatMessage.created_at, ChatMessage.id).all()
                )
                # 转换为 (user_message, assistant_message) 格式
                history = []
                sources = []
                i = 0
                while i < len(messages):
                    if messages[i].role == "user" and i + 1 < len(messages) and messages[i+1].role == "assistant":
                        history.append((messages[i].content, messages[i+1].content))
                        sources.append((messages[i+1].metadata_ or {}).get("sources", []))
                        i += 2
                    else:
                        i += 1
                return {"history": history, **({"sources": sources} if include_sources else {})}
            else:
                # 检查会话id是否存在
                existing_session = await db.run_sync(
                    lambda session: session.query(ChatSession).filter(ChatSession.id == session_id).first()
                )

                if existing_session:
                    # 会话存在但不属于当前用户
                    logger.warning(f"【数据库会话管理】会话 {session_id} 不属于用户 {user_id}")
                    from fastapi import HTTPException, status
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="当前会话不属于你"
                    )
                else:
                    # 会话不存在，创建一个新的
                    new_session = ChatSession(
                        id=session_id,
                        user_id=user_id,
                        title="新的对话"
                    )
                    db.add(new_session)
                    await db.commit()
                    await db.refresh(new_session)
                    logger.info(f"【数据库会话管理】创建新会话: {session_id} 属于用户: {user_id}")
                    return {"history": []}

    async def add_message(self, session_id: str, user_id: str, user_message: str, assistant_message: str, *, sources: list[dict] | None = None):
        """添加消息并保存到数据库"""
        async with AsyncSessionLocal() as db:
            # 检查会话id是否存在
            existing_session = await db.run_sync(
                lambda session: session.query(ChatSession).filter(ChatSession.id == session_id).first()
            )

            if existing_session:
                # 检查会话是否属于当前用户
                if existing_session.user_id != user_id:
                    # 会话存在但不属于当前用户，不添加消息
                    logger.warning(f"【数据库会话管理】会话 {session_id} 不属于用户 {user_id}，无法添加消息")
                    from fastapi import HTTPException, status
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="当前会话不属于你，无法添加消息"
                    )
                session = existing_session
            else:
                # 会话不存在，创建一个新的
                session = ChatSession(
                    id=session_id,
                    user_id=user_id,
                    title="新的对话"
                )
                db.add(session)
                await db.commit()
                await db.refresh(session)

            # 检查是否是新会话且标题为默认值，如果是则更新为用户的第一个问题
            if session.title == "新的对话":
                # 生成用户问题的摘要作为标题（截取前30个字符）
                title_summary = user_message[:30].strip()
                if len(user_message) > 30:
                    title_summary += "..."
                session.title = title_summary

            # 添加用户消息
            user_msg = ChatMessage(
                session_id=session.id,
                role="user",
                content=user_message
            )
            db.add(user_msg)

            # 添加助手消息
            assistant_msg = ChatMessage(
                session_id=session.id,
                role="assistant",
                content=assistant_message,
                metadata_={"sources": sources if sources is not None else (session.metadata_ or {}).get("pending_sources", [])},
            )
            db.add(assistant_msg)
            session.metadata_ = {key: value for key, value in (session.metadata_ or {}).items() if key != "pending_sources"}
            # 续聊时标题/metadata 可能完全不变，必须主动更新会话活动时间。
            session.updated_at = func.now()

            await db.commit()
            logger.info(f"【数据库会话管理】添加消息到会话: {session_id} 属于用户: {user_id}")

    async def get_history(self, session_id: str, user_id: str) -> list[tuple[str, str]]:
        """获取会话历史"""
        session_data = await self.get_session(session_id, user_id)
        return session_data.get("history", [])

    @staticmethod
    async def _check_access(db, session_id: str, user_id: str):
        owner = (await db.execute(select(ChatSession.user_id).where(ChatSession.id == session_id))).scalar_one_or_none()
        if owner is not None and owner != user_id:
            raise HTTPException(403, "当前会话不属于你")
        return owner is not None

    async def validate_session_access(self, session_id: str, user_id: str) -> None:
        """检索开始前校验所有权；不存在的会话交由后续正常消息创建。"""
        async with AsyncSessionLocal() as db:
            await self._check_access(db, session_id, user_id)

    async def get_recent_history(self, session_id: str, user_id: str, max_turns: int = 3) -> list[tuple[str, str]]:
        """只读最近完整问答，限制查询行数及正文长度，不加载来源快照或创建空会话。"""
        max_turns = min(3, max(0, max_turns))
        async with AsyncSessionLocal() as db:
            if not await self._check_access(db, session_id, user_id) or not max_turns:
                return []
            messages = list(reversed((await db.execute(select(ChatMessage.role, ChatMessage.content).where(
                ChatMessage.session_id == session_id).order_by(
                    ChatMessage.created_at.desc(), ChatMessage.id.desc()).limit(max_turns * 2))).all()))
        return [(messages[i].content[:600], messages[i + 1].content[:900])
                for i in range(len(messages) - 1)
                if messages[i].role == "user" and messages[i + 1].role == "assistant"]

    async def clear_session(self, session_id: str, user_id: str):
        """清除会话"""
        async with AsyncSessionLocal() as db:
            # 查找会话，验证属于该用户
            session = await db.run_sync(
                lambda session: session.query(ChatSession).filter(ChatSession.id == session_id, ChatSession.user_id == user_id).first()
            )

            if session:
                # 删除会话（级联删除消息）
                await db.delete(session)
                await db.commit()
                logger.info(f"【数据库会话管理】会话 {session_id} 已清除，属于用户: {user_id}")

    async def get_all_session_ids(self, user_id: str | None = None) -> list[str]:
        """获取所有会话 ID，如果提供了 user_id，则只返回该用户的会话"""
        async with AsyncSessionLocal() as db:
            if user_id:
                sessions = await db.run_sync(
                    lambda session: session.query(ChatSession).filter(ChatSession.user_id == user_id).all()
                )
            else:
                sessions = await db.run_sync(
                    lambda session: session.query(ChatSession).all()
                )
            return [session.id for session in sessions]

    async def get_user_sessions(self, user_id: str) -> list[dict]:
        """按最近消息时间排序并返回带 UTC 时区的时间；兼容旧会话未更新的 updated_at。"""
        latest_message_at = (
            select(func.max(ChatMessage.created_at))
            .where(ChatMessage.session_id == ChatSession.id)
            .correlate(ChatSession)
            .scalar_subquery()
        )
        activity_at = func.coalesce(latest_message_at, ChatSession.updated_at, ChatSession.created_at)
        async with AsyncSessionLocal() as db:
            sessions = (await db.execute(
                select(ChatSession, activity_at.label("activity_at"))
                .where(ChatSession.user_id == user_id)
                .order_by(activity_at.desc(), ChatSession.id)
            )).all()
            return [
                {
                    "id": session.id,
                    "title": session.title,
                    "pending_run_id": session.pending_run_id,
                    "created_at": _utc_isoformat(session.created_at),
                    "updated_at": _utc_isoformat(last_activity)
                }
                for session, last_activity in sessions
            ]

    async def get_pending_run_id(self, session_id: str, user_id: str) -> str | None:
        """返回该会话待审批 run_id；无会话或不属于该用户返回 None。"""
        async with AsyncSessionLocal() as db:
            session = await db.run_sync(
                lambda s: s.query(ChatSession)
                .filter(ChatSession.id == session_id, ChatSession.user_id == user_id)
                .first()
            )
            return session.pending_run_id if session else None

    async def set_pending_run_id(
        self, session_id: str, user_id: str, run_id: str | None, *, sources: list[dict] | None = None
    ) -> None:
        """写入/清除会话的待审批 run_id（校验所有权）。"""
        async with AsyncSessionLocal() as db:
            session = await db.run_sync(
                lambda s: s.query(ChatSession)
                .filter(ChatSession.id == session_id, ChatSession.user_id == user_id)
                .first()
            )
            if session is None:
                return
            session.pending_run_id = run_id
            if sources is not None:
                session.metadata_ = {**(session.metadata_ or {}), "pending_sources": sources}
            if run_id is None:
                session.metadata_ = {key: value for key, value in (session.metadata_ or {}).items() if key != "pending_sources"}
            await db.commit()


# 全局数据库会话管理器实例
database_session_manager = None

# 初始化数据库会话管理器
async def init_database_session_manager():
    """
    初始化数据库会话管理器
    :return: 初始化完成的 DatabaseSessionManager 实例
    """
    global database_session_manager
    database_session_manager = await DatabaseSessionManager.create()
    return database_session_manager
