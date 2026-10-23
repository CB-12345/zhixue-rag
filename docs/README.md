# 项目文档

## 开始使用与展示

| 文档 | 内容 |
| --- | --- |
| [项目 README](../README.md) | 功能、架构、配置与启动 |
| [GitHub 上传](GITHUB_PUBLISH.md) | 提交范围与上传步骤 |
| [最新收尾记录](PUBLICATION_READINESS.md) | 引用整理、目录清理、测试和剩余限制 |
| [五分钟演示](../demo/README.md) | 固定问题、原文定位、多轮问答与历史恢复 |
| [自编示例资料](../demo/ingestion/README.md) | 随仓库提供的 PDF、Word 和验收题 |
| [简历与演示讲稿](INTERNSHIP_READINESS.md) | 项目表达和讲解顺序 |
| [架构与面试](SYSTEM_AND_INTERVIEW.md) | 主要模块及技术选择 |

## RAG 实现

| 文档 | 内容 |
| --- | --- |
| [结构化导入](STRUCTURED_INGESTION.md) | 标题、表格、页码与切片 |
| [索引与图谱解耦](INDEX_GRAPH_DECOUPLING.md) | 异步索引、失败与重试 |
| [文档生命周期](DOCUMENT_LIFECYCLE.md) | 上传身份、去重、替换与删除 |
| [重排序](RERANK_AND_EVALUATION.md) | 模型配置、候选与检索指标 |
| [检索隔离](SCOPED_VECTOR_RETRIEVAL.md) | 用户与来源过滤 |
| [多轮追问](CONVERSATIONAL_RETRIEVAL.md) | 指代补全及历史边界 |
| [原文定位](SOURCE_LOCATION.md) | 引用、切片与历史快照 |
| [可答性检查](EVIDENCE_GATING.md) | 证据不足和生成前拒答 |
| [最终回答核对](FINAL_ANSWER_VERIFICATION.md) | 事实、引用与逐字摘录 |
| [摘录修复](ANSWER_QUOTE_REPAIR.md) | 有限重试及原始失败案例 |

## 代表性实验

- [Top-K 对比](topk-20261023/README.md)：同一候选池比较 3、5、8 条上下文；保留开发集、验证集和失败记录。
- [真实资料评测](REAL_MATERIAL_EVALUATION.md)：用户课程资料的问答、代码与拒答边界。
- [回答质量](ANSWER_QUALITY_EVALUATION.md)：区分关键词检查、核心要点和引用支持。
- [性能与稳定性](PERFORMANCE_AND_STABILITY.md)：SSE 耗时、取消与超时。

其他按日期命名的目录保存历史实验及问题诊断，用于追溯当时的输入、输出和限制。它们不会参与应用运行，也不能替代最新版本的验收结果。
