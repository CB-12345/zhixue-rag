# 本地开发

## 环境

- Python 3.12、uv。
- Node.js 22、npm。
- MySQL 8、Redis 7、Neo4j 5.26；完整运行可通过 Docker Compose 启动。

优先按 [README](../README.md) 配置并运行整套服务。只修改前端时，可以复用已经运行的后端。

## 前端

```bash
cd front
npm ci
npm run dev
```

开发服务器的 API 代理配置见 [vite.config.ts](../front/vite.config.ts)，后端默认地址为 `http://localhost:8000`。

```bash
npm run lint
npm run build
```

生产镜像构建静态资源并使用 Nginx 提供页面和 API 代理。SSE 路径关闭代理缓冲，页面路由与 API 路径分别配置。

## 后端

```bash
cd backend
uv sync --frozen --extra dev --no-group parsers
uv run --no-sync ruff check .
uv run --no-sync pytest -q
```

上述安装模式用于常规代码检查与测试，省略较大的文档解析依赖。实际执行多格式解析时安装默认依赖组：

```bash
uv sync --frozen --extra dev
uv run --no-sync uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

本机运行后端需要为 `backend/.env` 配置可访问的基础设施地址。Compose 内使用服务名，本机开发进程需要使用实际可访问的地址与端口；默认 Compose 不向宿主机开放数据库端口。

也可以用开发覆盖文件挂载后端代码：

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --build
```

## 代码导航

- `backend/main.py`：应用入口和路由注册。
- `backend/app/router/`：HTTP 接口与问答服务。
- `backend/app/rag/`：文档处理、切片、检索与回答核对。
- `backend/app/graph/`：索引任务、图存储和扩展关系检索。
- `backend/app/services/`：笔记、会话等业务服务。
- `backend/app/core/settings.py`：配置字段、默认值与参数约束。
- `front/src/`：前端页面、组件、流式请求和来源查看。

## 常见排查

| 现象 | 优先检查 |
| --- | --- |
| 文档无法检索 | 基础索引是否完成、Embedding 是否可用、Neo4j 索引状态、当前用户范围 |
| 生成前提示资料不足 | 检索证据是否包含所问信息、切片是否保留关键行、问题是否超出资料范围 |
| 提示暂时无法完成核对 | 模型服务响应、核对阶段超时和网络情况 |
| 重排没有生效 | provider、基础地址、模型权限及 Key 是否匹配 |
| 修改配置后仍使用旧值 | 是否重新创建后端容器 |
| 重建后页面 API 返回 502 | 后端健康状态；必要时重启前端刷新代理连接 |
| 历史可以看引用但原文打不开 | 当前文档是否已删除或更新、来源类型及原文接口是否可用 |

查看服务状态和日志：

```bash
docker compose ps
docker compose logs --tail 100 backend
```

运行日志、真实模型输出、用户资料和实际环境配置应保存在已忽略的本机目录中。
