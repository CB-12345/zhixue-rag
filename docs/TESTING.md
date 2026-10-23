# 测试与验证工具

## 自动化测试

后端测试覆盖鉴权、文档登记、任务状态、权限隔离、检索与重排协议、原文引用、回答核对、取消与超时、SSE 和会话恢复等路径。

```bash
cd backend
uv sync --frozen --extra dev --no-group parsers
uv run --no-sync ruff check .
uv run --no-sync pytest -q
```

常规测试使用替身、临时文件和隔离数据库，不要求真实模型 API。依赖外部数据库的集成测试需要单独的测试服务；未配置时会跳过，不能按通过统计。

前端检查：

```bash
cd front
npm ci
npm run lint
npm run build
```

仓库 CI 在 `main`、`master` 的 push 和 Pull Request 上运行上述检查。常规 CI 不配置模型密钥。

## 演示资料与测试数据

- `demo/documents/`：简短 Markdown 资料，可快速体验上传和跨文档问答。
- `demo/ingestion/`：自编 PDF、DOCX 与对应问题，覆盖表格、章节、页码及资料不足。
- `backend/evaluation/`：自编的检索、回答及核对测试输入。
- `backend/tests/fixtures/`：精简的结构与协议回归样例。

题集是功能验证材料，不能直接用于宣称通用回答准确率。关键词命中、引用编号合法、答案受到证据支持属于不同检查目标。

## 可选的真实 API 验证

以下工具会调用所配置的服务，可能产生模型费用。运行前先准备配置和测试环境。

### 演示资料问答

在项目根目录执行，密码通过交互输入：

```bash
python scripts/acceptance.py --seed
```

工具使用登录账号上传演示资料并创建测试会话，检查 SSE 完成、关键词、引用与历史恢复。默认输出位于 `backend/data/evaluation/acceptance-latest.json`。这是小型流程验收，答案语义仍需复核。

### 检索与重排

在已安装后端依赖且基础设施可访问的环境执行：

```bash
cd backend
uv run --no-sync python scripts/eval_fixed.py
```

工具使用独立的随机评测空间导入公开测试输入，对比检索与重排并清理对应测试数据。默认结果写入 `backend/results`。

其他脚本可以分别验证可答性、回答生成、最终核对及实际文件入库。具体参数使用对应脚本的 `--help`；涉及真实数据库时使用独立测试环境。

### SSE 性能

```bash
python scripts/benchmark_rag.py --help
```

工具区分首个事件、首段正式回答与完整结束耗时。由于资料回答会先生成并核对，首个处理事件和首段正文代表不同阶段。

真实运行报告、诊断跟踪和带来源正文的输出保存在本机已忽略目录中。公开仓库提供工具与输入，不存放个人运行过程文件。
