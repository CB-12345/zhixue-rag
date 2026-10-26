# 自动化检查

## 后端

测试覆盖鉴权与权限、文档生命周期、后台任务、结构切片、检索、重排协议、引用核对、会话恢复、审批、超时和取消等路径。

~~~bash
cd backend
uv sync --frozen --extra dev --no-group parsers
uv run --no-sync ruff check .
uv run --no-sync pytest -q
~~~

常规测试使用隔离数据库、临时文件和替身，不要求真实模型 API。外部服务集成测试需要独立测试服务，未配置时会跳过；跳过不计为通过。多格式解析实际运行需安装默认 parsers 依赖组，参见 [开发说明](DEVELOPMENT.md)。

## 前端

~~~bash
cd front
npm ci
npm run lint
npm run build
~~~

## 持续集成

GitHub Actions 在 main、master 的 push 与 Pull Request 上执行后端静态检查及测试、前端检查与生产构建。常规 CI 不配置真实模型密钥。

## 手动体验

使用 [自编示例资料](../demo/README.md)，检查上传、基础索引、问答、引用、追问、笔记审批及历史恢复。

模型答案需要结合原文复核，接口完成和引用编号合法不等于答案语义正确。实际凭据、用户资料、运行日志和临时诊断输出保存在本机忽略目录。
