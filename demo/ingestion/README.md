# 实际文件上传验收夹具

这两份文件由助手自编，课程、编号、姓名及规则均为虚构。用于复现文件解析、长章节和表格切片、后台索引、问答与来源页码；不是外部课程材料或公开准确率基准。

- `xinghe-lab-handbook.pdf`：3 个物理页；第 1 页长正文，第 2、3 页为规则网格表格，共 50 条设备记录。
- `xinghe-course-rules.docx`：标题分节、超过单片预算的正文、38 条申请记录表格；DOCX 不提供推算页码。
- `manifest.json`：首次真实问答前固定的文件 SHA-256、8 道题、参考要点、原文依据和必需引用页码。不要按回答修改题集后再声称首次结果。

在项目根目录执行，复用本机已配置的模型 API；会产生模型调用费用：

```powershell
docker cp demo/ingestion/. ragnotebook-backend-1:/app/data/evaluation/ingestion-fixtures
docker exec ragnotebook-backend-1 python scripts/eval_ingestion.py
```

脚本默认通过 `http://frontend` 调用上传与 SSE 问答，创建随机临时资料空间，不注册账号；完成后删除测试会话、资料与任务，并核对原有文档登记和切片正文哈希。每次使用新输出目录，结果位于 `backend/data/evaluation/ingestion-*`。此工具需要后端容器的本机管理权限，不是开放给普通用户的认证接口。

关键词、来源和页码检查通过后仍须审阅整段回答：引用编号存在不等于句子受到原文支持。实际结果与已知限制见[结构化导入记录](../../docs/STRUCTURED_INGESTION.md)。
