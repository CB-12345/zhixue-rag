# 模型与运行配置

配置模板为 [backend/.env.example](../backend/.env.example)，复制到 `backend/.env` 后填写。密钥、数据库和上传资料保存在本机。

## 对话与嵌入

| 变量 | 用途 |
| --- | --- |
| `OPENAI_BASE_URL` / `OPENAI_API_KEY` / `OPENAI_MODEL_NAME` | OpenAI 兼容的对话服务 |
| `EMBED_BASE_URL` / `EMBED_API_KEY` / `EMBED_MODEL_NAME` | 生成文档与问题向量 |
| `PLANNER_BASE_URL` / `PLANNER_API_KEY` / `PLANNER_MODEL_NAME` | 可选的独立规划模型 |
| `SECRET_KEY` | JWT 签名，使用自行生成的随机值 |

对话模型与嵌入模型可以来自不同供应商，分别填写完整的地址、密钥和模型名。服务地址通常包含供应商要求的 `/v1` 前缀；模型名应与对应服务实际支持的名称一致。

未单独配置规划服务时，系统按配置回退规则复用现有通道。对话与嵌入来自不同平台时，建议显式填写 `PLANNER_*` 三项，避免规划模型名与服务不匹配。

更换嵌入模型可能改变向量维度和语义空间。已有资料需要按新模型重新建立索引，不能直接把不同模型产生的向量混用。

## 重排序

### 百炼 DashScope

```dotenv
RERANKER_ENABLED=true
RERANKER_PROVIDER=dashscope
RERANKER_API_BASE_URL=https://dashscope.aliyuncs.com/api/v1
RERANKER_API_KEY=your-rerank-api-key
RERANKER_MODEL=your-enabled-rerank-model
```

接口会在基础地址后拼接 `/services/rerank/text-rerank/text-rerank`，发送 `input`、`parameters` 并读取 `output.results`。这里使用百炼原生协议地址。

如果模型控制台提供工作空间专属地址，使用该空间对应的基础地址，例如 `https://<workspace-id>.cn-beijing.maas.aliyuncs.com/api/v1`。地域、工作空间、Key 和已开通模型需要对应；基础地址不要重复包含完整的重排路径。

### Compatible 接口

```dotenv
RERANKER_ENABLED=true
RERANKER_PROVIDER=compatible
RERANKER_API_BASE_URL=https://api.siliconflow.cn/v1
RERANKER_API_KEY=your-rerank-api-key
RERANKER_MODEL=your-enabled-rerank-model
```

此模式请求 `{base_url}/rerank`，使用 `model`、`query`、`documents` 字段，并要求返回含 `index`、`relevance_score` 的 `results`。供应商需支持这一请求与响应结构。

没有重排服务时可以设置 `RERANKER_ENABLED=false`。未配置或调用失败时按原检索顺序继续处理，并执行后续证据核对。

## 检索与等待预算

| 变量 | 默认值 | 含义 |
| --- | --- | --- |
| `RERANKER_CANDIDATE_K` | 20 | 重排启用且配置完整时，每个文本检索步骤的最低候选数 |
| `RAG_CONTEXT_K` | 3 | 最终选取的本地证据数量，范围 1～8 |
| `RERANKER_TIMEOUT_SECONDS` | 8 | 重排等待上限 |
| `ANSWERABILITY_TIMEOUT_SECONDS` | 8 | 生成前的可答性核对预算 |
| `ANSWER_VERIFICATION_ENABLED` | true | 是否执行最终回答核对 |
| `ANSWER_VERIFICATION_TIMEOUT_SECONDS` | 12 | 最终回答核对及有限修复的共同预算 |
| `QUERY_REWRITE_TIMEOUT_SECONDS` | 4 | 多轮问题补全等待上限 |
| `GRAPH_RETRIEVAL_TIMEOUT_SECONDS` | 4 | 补充图谱检索等待上限 |
| `AGENT_RUN_TIMEOUT_SECONDS` | 60 | Agent 执行阶段预算，不包含前置 RAG |
| `VECTOR_EXACT_SCOPE_LIMIT` | 2000 | 使用精确向量排序的最大来源范围 |
| `VECTOR_ANN_MAX_CANDIDATES` | 2048 | 大范围 ANN 候选扩展上限 |

默认准备证据的字符预算为 6000，融合后最多 8 条证据。局部片段数、候选数与字符预算是不同约束。扩大 Top-K 会增加上下文长度和核对成本，需要结合实际资料选择。

视觉解析与网页补证属于可选能力，分别由 `VISION_ENABLED`、`WEB_SEARCH_ENABLED` 控制，默认关闭。明确要求仅依据本地资料的问题不会通过网页补证绕过资料范围。

## Docker 与持久化

`docker-compose.yml` 为后端注入 MySQL、Redis 和 Neo4j 的容器地址。模型配置从 `backend/.env` 读取；基础设施凭据由 Compose 统一注入。

根目录 `.env` 可以覆盖 Compose 中的 `MYSQL_ROOT_PASSWORD` 和 `NEO4J_PASSWORD`。已有数据库卷修改密码还需要按数据库自身流程更新凭据，不能只修改配置文件。

模型配置修改后执行：

```bash
docker compose up -d --force-recreate backend
docker compose restart frontend
```

数据库使用 Docker volumes，媒体与运行目录使用本机挂载。`docker compose down` 保留数据卷；删除数据卷前应自行备份资料。
